// 包含标准输入输出（调试打印）
#include <iostream>
// 包含自定义的TensorRT接口（封装GPU推理、内存管理）
#include "NvidiaInterface.hpp"
// 包含分类模块头文件（定义Infer基类、Image结构体等）
#include "classify.hpp"

// 分类模型推理的命名空间（隔离分类相关逻辑）
namespace classify {

// 引入标准命名空间（简化代码，如std::vector→vector）
using namespace std;

/**
 * @brief 分类模型后处理函数：从模型输出中获取概率最大的类别索引
 * @param output_array 模型输出的类别概率数组（长度=类别数，每个元素对应一个类别的置信度/概率）
 * @return 最大概率对应的类别索引（即模型预测的类别）
 */
int postprocess(vector<float>& output_array)
{
    int   max_index = 0;    // 初始化最大概率的类别索引（默认第0类）
    float max_value = output_array[0];  // 初始化最大概率值（默认第0类的概率）

    // 遍历所有类别，找到最大概率对应的索引
    for (int i = 1; i < output_array.size(); i++) {
        if (output_array[i] > max_value) {
            max_value = output_array[i];  // 更新最大概率值
            max_index = i;                // 更新最大概率的类别索引
        }
    }
    return max_index;  // 返回预测类别索引
}

/**
 * @brief 内联函数：将数值向上对齐到指定的内存边界（默认32字节）
 * @param n 待对齐的原始数值（如内存大小、数组长度）
 * @param align 对齐边界（默认32，GPU内存访问需对齐以提升效率）
 * @return 对齐后的数值（向上取整到align的整数倍）
 * 计算逻辑：(n + align - 1) / align * align → 确保结果≥n且是align的倍数
 */
inline int upbound(int n, int align = 32)
{
    return (n + align - 1) / align * align;
}

/**
 * @brief 分类推理核心实现类（继承自Infer<int>，泛型参数int表示输出为类别索引）
 * 功能：封装TensorRT引擎加载、图像预处理、批量推理、结果后处理全流程
 */
class InferImpl : public Infer<int> {
public:
    // -------------------------- 成员变量说明 --------------------------
    shared_ptr<trt::Infer>                         trt_;  // TensorRT推理实例（封装引擎加载、前向传播）
    string                                         engine_file_;  // TensorRT引擎文件路径（.engine）
    Type                                           type_;  // 分类模型类型（如densenet121）
    // 预处理缓存：每个批次图像对应一个缓存（存储CPU/GPU上的临时数据，避免重复分配）
    vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;
    trt::Memory<float> input_buffer_;   // 模型输入内存（CPU+GPU，存储预处理后的图像数据）
    trt::Memory<float> output_array_;   // 模型输出内存（CPU+GPU，存储类别概率数组）
    int                num_class_ = 0;  // 分类任务的类别总数（如10类、1000类）
    int                network_input_width_;  // 模型输入图像宽度（如224）
    int                network_input_height_; // 模型输入图像高度（如224）
    bool               isdynamic_model_ = false;  // 是否为动态批次模型（支持可变输入批次）
    Norm               normalize_;  // 图像归一化配置（均值标准差/线性缩放）
    float              mean_[3], std_[3];  // 归一化用的均值、标准差（如ImageNet数据集的均值）

    // 虚析构函数：确保子类析构时正确释放父类资源
    virtual ~InferImpl() = default;

    /**
     * @brief 根据批次大小调整输入/输出内存及预处理缓存
     * @param batch_size 当前推理的批次大小（如1、4、8）
     * 作用：避免内存浪费，按需分配GPU/CPU内存，确保内存足够存储批次数据
     */
    void adjust_memory(int batch_size)
    {
        // 计算单张输入图像的元素总数：宽 × 高 × 3通道（RGB/BGR）
        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        // 分配输入内存：GPU端分配 batch_size × input_numel 个float元素（预处理后的数据为float）
        input_buffer_.gpu(batch_size * input_numel);
        // 分配输出内存：GPU端存储概率数组，CPU端用于接收GPU数据（后处理）
        output_array_.gpu(batch_size * num_class_);
        output_array_.cpu(batch_size * num_class_);

        // 调整预处理缓存数量：确保缓存数量≥批次大小（每个图像用一个缓存）
        if ((int)preprocess_buffers_.size() < batch_size) {
            for (int i = preprocess_buffers_.size(); i < batch_size; ++i) {
                // 每个缓存同时管理CPU和GPU内存（用于存储图像原始数据和仿射矩阵）
                preprocess_buffers_.push_back(make_shared<trt::Memory<unsigned char>>());
            }
        }
    }

    /**
     * @brief 单张图像的预处理（CPU→GPU数据传输、仿射变换、归一化）
     * @param ibatch 当前图像在批次中的索引（0~batch_size-1）
     * @param image 输入图像（自定义Image结构体，包含BGR数据、宽高）
     * @param preprocess_buffer 该图像的预处理缓存（CPU+GPU）
     * @param affine 仿射矩阵（用于图像缩放居中，后续无需反向映射，分类无需返回坐标）
     * @param stream CUDA流（用于异步操作，提升效率）
     * 核心流程：1.计算仿射矩阵 2.CPU→GPU数据拷贝 3.仿射变换+归一化（CUDA核函数）
     */
    void preprocess(int ibatch, const Image& image,
                   shared_ptr<trt::Memory<unsigned char>> preprocess_buffer,
                   AffineMatrix& affine, void* stream = nullptr)
    {
        // 1. 计算仿射矩阵：将原始图像缩放到模型输入尺寸，保持宽高比并居中（避免拉伸）
        affine.compute(
            make_tuple(image.width, image.height),  // 原始图像尺寸（宽，高）
            make_tuple(network_input_width_, network_input_height_)  // 模型输入尺寸
        );

        // 计算内存大小：输入图像元素总数、仿射矩阵内存（对齐后）
        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        size_t size_image = image.width * image.height * 3;  // 单张图像字节数（uint8×3通道）
        size_t size_matrix = upbound(sizeof(affine.d2i), 32);  // 仿射矩阵内存（32字节对齐）

        // 2. 分配GPU工作空间：存储仿射矩阵（float）和原始图像（uint8）
        uint8_t* gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);
        float*   affine_matrix_device = (float*)gpu_workspace;  // GPU上的仿射矩阵地址
        uint8_t* image_device = gpu_workspace + size_matrix;    // GPU上的图像数据地址

        // 分配CPU工作空间：暂存图像数据和仿射矩阵（准备拷贝到GPU）
        uint8_t* cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
        float*   affine_matrix_host = (float*)cpu_workspace;  // CPU上的仿射矩阵地址
        uint8_t* image_host = cpu_workspace + size_matrix;    // CPU上的图像数据地址

        // 转换CUDA流（输入为void*，需转为cudaStream_t）
        cudaStream_t stream_ = (cudaStream_t)stream;

        // 3. CPU→CPU内存拷贝：将原始图像数据（image.bgrptr）和仿射矩阵（affine.d2i）拷到工作空间
        std::chrono ::high_resolution_clock::time_point a1 = std::chrono::high_resolution_clock::now();
        memcpy(image_host, image.bgrptr, size_image);          // 拷贝图像数据（BGR格式）
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));  // 拷贝仿射矩阵
        std::chrono ::high_resolution_clock::time_point a2 = std::chrono::high_resolution_clock::now();
        auto time_used2 = std::chrono::duration_cast<std::chrono::duration<double>>(a2 - a1);

        // 4. CPU→GPU异步拷贝：使用CUDA流异步传输，避免阻塞CPU
        checkRuntime(cudaMemcpyAsync(
            image_device, image_host, size_image,
            cudaMemcpyHostToDevice, stream_  // 主机到设备（CPU→GPU）
        ));
        checkRuntime(cudaMemcpyAsync(
            affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
            cudaMemcpyHostToDevice, stream_
        ));

        // 5. CUDA核函数：执行双线性插值仿射变换（缩放图像）+ 归一化（减均值/除标准差）
        // 输入：GPU上的原始图像；输出：GPU上的预处理后数据（input_device）
        warp_affine_bilinear_and_normalize_plane(
            image_device, image.width * 3,  // 输入图像步长（每行字节数：宽×3通道）
            image.width, image.height,      // 原始图像宽高
            input_buffer_.gpu() + ibatch * input_numel,  // 当前批次输入的GPU地址
            network_input_width_, network_input_height_,  // 模型输入宽高
            affine_matrix_device, 114,      // 仿射矩阵、填充值（114，避免黑边影响）
            normalize_, stream_             // 归一化配置、CUDA流
        );
    }

    /**
     * @brief 加载TensorRT引擎并初始化推理参数
     * @param engine_file TensorRT引擎文件路径（.engine，提前由ONNX转换生成）
     * @param type 分类模型类型（如densenet121，标识模型结构）
     * @return true-加载成功，false-加载失败
     * 核心步骤：加载引擎→解析输入输出维度→初始化归一化参数→设置类别数
     */
    bool load(const string& engine_file, Type type)
    {
        // 1. 加载TensorRT引擎实例（trt::load封装了引擎反序列化）
        trt_ = trt::load(engine_file);
        if (trt_ == nullptr) {  // 引擎加载失败（如文件不存在、格式错误）
            return false;
        }

        // 2. 打印引擎信息（如输入输出维度、层信息，调试用）
        trt_->print();

        // 3. 记录模型类型和引擎文件路径
        this->type_ = type;
        this->engine_file_ = engine_file;

        // 4. 获取模型输入/输出维度（静态批次模型）
        auto input_dim = trt_->static_dims(0);    // 输入维度：[batch, channel, height, width]（TensorRT默认NCHW）
        auto output_dims_ = trt_->static_dims(1); // 输出维度：[batch, num_class]

        // 调试打印输入维度（验证是否符合预期）
        puts("input_dim");
        for (int i = 0; i < 4; i++) {
            std::cout << input_dim[i] << std::endl;
        }

        // 5. 提取模型输入尺寸（height=input_dim[2]，width=input_dim[3]，NCHW格式）
        network_input_width_ = input_dim[3];
        network_input_height_ = input_dim[2];

        // 6. 判断是否为动态批次模型（TensorRT动态维度需提前配置）
        isdynamic_model_ = trt_->has_dynamic_dim();

        // 7. 初始化归一化参数（使用ImageNet数据集的均值标准差，适配预训练模型）
        mean_[0] = 0.485;  // B通道均值（因后续SwapRB，实际对应模型的R通道）
        mean_[1] = 0.456;  // G通道均值
        mean_[2] = 0.406;  // R通道均值
        std_[0] = 0.229;   // B通道标准差
        std_[1] = 0.224;   // G通道标准差
        std_[2] = 0.225;   // R通道标准差
        // 配置归一化：均值标准差模式，缩放因子1/255（将uint8[0,255]转为float[0,1]），通道交换（BGR→RGB）
        normalize_ = Norm::mean_std(mean_, std_, 1.0 / 255.0, ChannelType::SwapRB);

        // 8. 提取类别数（输出维度的第二个元素：output_dims_[1] = num_class）
        num_class_ = output_dims_[1];

        return true;  // 加载成功
    }

    /**
     * @brief 单张图像推理（重写父类纯虚函数）
     * @param image 输入图像（Image结构体）
     * @param stream CUDA流（默认nullptr，使用默认流）
     * @return 预测的类别索引（int）
     * 逻辑：调用批量推理接口forwards处理单张图像，返回第一个结果
     */
    virtual int forward(const Image& image, void* stream = nullptr) override
    {
        auto output = forwards({image}, stream);  // 批量推理（批次大小1）
        if (output.empty()) {                     // 推理失败（如无输出）
            return {};
        }
        return output[0];  // 返回单张图像的类别索引
    }

    /**
     * @brief 批量图像推理（重写父类纯虚函数，核心推理接口）
     * @param images 输入图像列表（vector<Image>）
     * @param stream CUDA流（用于异步操作）
     * @return 批量图像的预测类别索引列表（vector<int>）
     * 核心流程：批次检查→内存调整→批量预处理→TensorRT前向→结果拷贝→后处理
     */
    virtual vector<int> forwards(const vector<Image>& images, void* stream = nullptr) override
    {
        int num_image = images.size();  // 当前推理的图像数量（批次大小）
        if (num_image == 0) {           // 无图像输入，直接返回空列表
            return {};
        }

        // 1. 处理批次大小（适配静态/动态批次模型）
        auto input_dims = trt_->static_dims(0);  // 静态输入维度（NCHW）
        int  infer_batch_size = input_dims[0];   // 初始批次大小（静态模型的固定批次）

        if (infer_batch_size != num_image) {  // 当前图像数量≠初始批次大小
            if (isdynamic_model_) {           // 动态批次模型：调整批次大小
                infer_batch_size = num_image;
                input_dims[0] = num_image;    // 更新输入维度的批次值
                // 设置TensorRT动态批次（失败则返回空）
                if (!trt_->set_run_dims(0, input_dims)) {
                    return {};
                }
            } else {                          // 静态批次模型：检查图像数量是否≤最大批次
                if (infer_batch_size < num_image) {
                    INFO("When using static shape model, number of images[%d] must be less than or equal to the maximum batch[%d].",
                         num_image, infer_batch_size);
                    return {};
                }
            }
        }

        // 2. 调整内存：根据当前批次大小分配输入/输出及预处理缓存
        std::chrono ::high_resolution_clock::time_point a1 = std::chrono::high_resolution_clock::now();
        adjust_memory(infer_batch_size);
        std::chrono ::high_resolution_clock::time_point a2 = std::chrono::high_resolution_clock::now();
        auto time_used2 = std::chrono::duration_cast<std::chrono::duration<double>>(a2 - a1);

        // 3. 批量预处理：为每张图像计算仿射矩阵并执行预处理（CPU→GPU+变换+归一化）
        vector<AffineMatrix> affine_matrixs(num_image);  // 存储每张图像的仿射矩阵
        std::chrono ::high_resolution_clock::time_point a3 = std::chrono::high_resolution_clock::now();
        cudaStream_t stream_ = (cudaStream_t)stream;
        for (int i = 0; i < num_image; ++i) {
            // 调用preprocess处理第i张图像，使用对应的预处理缓存
            preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);
        }

        // 4. 准备TensorRT前向传播的绑定数据（输入输出的GPU地址）
        vector<void*> bindings{input_buffer_.gpu(), output_array_.gpu()};

        // 5. 执行TensorRT前向传播（推理）
        std::chrono ::high_resolution_clock::time_point a3d1 = std::chrono::high_resolution_clock::now();
        auto time_used3 = std::chrono::duration_cast<std::chrono::duration<double>>(a3d1 - a3);
        if (!trt_->forward(bindings, stream)) {  // 前向失败（如引擎错误、内存不足）
            INFO("Failed to tensorRT forward.");
            return {};
        }
        std::chrono ::high_resolution_clock::time_point a3d2 = std::chrono::high_resolution_clock::now();
        auto time_used3d2 = std::chrono::duration_cast<std::chrono::duration<double>>(a3d2 - a3d1);

        // 6. GPU→CPU异步拷贝：将模型输出（概率数组）从GPU拷到CPU（后处理需CPU计算）
        std::chrono ::high_resolution_clock::time_point a3d3 = std::chrono::high_resolution_clock::now();
        auto time_used3d3 = std::chrono::duration_cast<std::chrono::duration<double>>(a3d3 - a3d2);
        checkRuntime(cudaMemcpyAsync(
            output_array_.cpu(), output_array_.gpu(),
            output_array_.gpu_bytes(),  // 拷贝的字节数（批次大小×类别数×sizeof(float)）
            cudaMemcpyDeviceToHost, stream_  // 设备到主机（GPU→CPU）
        ));
        checkRuntime(cudaStreamSynchronize(stream_));  // 等待拷贝完成（同步CUDA流）
        std::chrono ::high_resolution_clock::time_point a4 = std::chrono::high_resolution_clock::now();
        auto time_used3d4 = std::chrono::duration_cast<std::chrono::duration<double>>(a4 - a3d3);

        // 打印前向+解码耗时（调试性能用）
        auto time_used4 = std::chrono::duration_cast<std::chrono::duration<double>>(a4 - a3);
        INFO("forward and decode_kernel_invoker time: %f", time_used4.count() * 1000);

        // 7. 批量后处理：为每张图像计算预测类别
        std::chrono ::high_resolution_clock::time_point a5 = std::chrono::high_resolution_clock::now();
        vector<int> arrout(num_image);  // 存储最终的类别索引列表
        for (int ib = 0; ib < num_image; ++ib) {
            // 获取当前批次图像的输出概率数组（CPU端）
            float*        parray = output_array_.cpu() + ib * num_class_;
            vector<float> output(num_class_);  // 转换为vector便于后处理
            for (int o = 0; o < num_class_; o++) {
                output[o] = parray[o];
            }
            // 调用postprocess获取最大概率对应的类别索引
            arrout[ib] = postprocess(output);
        }

        return arrout;  // 返回批量图像的类别索引列表
    }
};

/**
 * @brief 创建分类推理实例（返回原始指针）
 * @param engine_file TensorRT引擎文件路径
 * @param type 分类模型类型
 * @return Infer<int>* 推理实例指针（失败返回nullptr）
 * 作用：内部创建InferImpl实例，加载引擎，供load函数封装为智能指针
 */
Infer<int>* loadraw(const std::string& engine_file, Type type)
{
    InferImpl* impl = new InferImpl();  // 创建核心实现类实例
    // 加载引擎失败：删除实例，返回nullptr
    if (!impl->load(engine_file, type)) {
        delete impl;
        impl = nullptr;
    }
    return impl;  // 返回实例指针
}

/**
 * @brief 创建分类推理实例（返回智能指针，推荐外部调用）
 * @param engine_file TensorRT引擎文件路径
 * @param type 分类模型类型
 * @return shared_ptr<Infer<int>> 智能指针（自动管理内存，避免内存泄漏）
 * 作用：封装loadraw的原始指针为shared_ptr，简化外部使用并确保内存安全
 */
shared_ptr<Infer<int>> load(const string& engine_file, Type type)
{
    // 将loadraw返回的原始指针转换为InferImpl指针，封装为shared_ptr
    return std::shared_ptr<InferImpl>((InferImpl*)loadraw(engine_file, type));
}

}  // namespace classify
