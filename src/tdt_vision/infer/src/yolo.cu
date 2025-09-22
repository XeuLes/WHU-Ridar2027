// 保持原头文件引用顺序，不修改
#include "BaseInfer.hpp"
#include "NvidiaInterface.hpp"
#include "yolos.hpp"

// 保留原命名空间
namespace yolo {

using namespace std;
using namespace tdt_radar;

// 保留原静态常量定义
static const int point_num = 4;
const int NUM_BOX_ELEMENT = 15;
const int MAX_IMAGE_BOXES = 1024;

// 保持原内联函数逻辑，优化代码换行
inline int upbound(int n, int align = 32)
{
    return (n + align - 1) / align * align;
}

// 保留原仿射投影函数（主机/设备端通用），拆分长参数列表
static __host__ __device__ void affine_project(
    float* matrix, 
    float x, 
    float y, 
    float* ox, 
    float* oy
)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

// 保留原通用YOLO解码核函数，优化参数换行与代码缩进
static __global__ void decode_kernel_common(
    float* predict, 
    int num_bboxes, 
    int num_classes,
    int output_cdim, 
    float confidence_threshold,
    float* invert_affine_matrix, 
    float* parray,
    int MAX_IMAGE_BOXES
)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    float* pitem = predict + output_cdim * position;

    // 目标存在性置信度筛选
    float objectness = pitem[4];
    if (objectness < confidence_threshold)
        return;

    // 筛选最大类别置信度
    float* class_confidence = pitem + 5;
    float  confidence = *class_confidence++;
    int    label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }

    // 最终置信度计算与筛选
    confidence *= objectness;
    if (confidence < confidence_threshold)
        return;

    // 原子操作计数有效检测框
    int index = atomicAdd(parray, 1);
    if (index >= MAX_IMAGE_BOXES)
        return;

    // 解码边界框（cx,cy,w,h → left,top,right,bottom）
    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    // 映射边界框到原始图像尺寸
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    // 写入输出检测框数组
    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1=保留，0=忽略
}

// 保留原YOLOV5Face解码核函数，优化参数与代码块缩进
static __global__ void decode_kernel_v5_face(
    float* predict, 
    int num_bboxes, 
    int num_classes,
    int output_cdim, 
    float confidence_threshold,
    float* invert_affine_matrix, 
    float* parray,
    int MAX_IMAGE_BOXES
)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    float* pitem = predict + output_cdim * position;

    // 目标存在性置信度筛选
    float objectness = pitem[4];
    if (objectness < confidence_threshold)
        return;

    // 筛选最大类别置信度（跳过关键点通道）
    float* class_confidence = pitem + 5 + 2 * point_num;
    float  confidence = *class_confidence++;
    int    label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }

    // 最终置信度计算与筛选
    confidence *= objectness;
    if (confidence < confidence_threshold)
        return;

    // 原子操作计数有效检测框
    int index = atomicAdd(parray, 1);
    if (index >= MAX_IMAGE_BOXES)
        return;

    // 解码边界框
    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    objectness = *pitem++;  // 跳过已使用的objectness
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    // 解码关键点（4个点，每个点含x/y坐标）
    float landmark_array[point_num * 2];
    for (int i = 0; i < point_num; i++) {
        landmark_array[2 * i] = *pitem++;
        landmark_array[2 * i + 1] = *pitem++;
    }

    // 映射边界框与关键点到原始图像尺寸
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);
    for (int i = 0; i < point_num; i++) {
        affine_project(
            invert_affine_matrix, 
            landmark_array[2 * i],
            landmark_array[2 * i + 1], 
            &landmark_array[2 * i],
            &landmark_array[2 * i + 1]
        );
    }

    // 写入输出（含关键点）
    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1=保留，0=忽略
    for (int i = 0; i < point_num; i++) {
        *pout_item++ = landmark_array[2 * i];
        *pout_item++ = landmark_array[2 * i + 1];
    }
}

// 保留原YOLOV8解码核函数，优化代码换行
static __global__ void decode_kernel_v8(
    float* predict, 
    int num_bboxes, 
    int num_classes,
    int output_cdim, 
    float confidence_threshold,
    float* invert_affine_matrix, 
    float* parray,
    int MAX_IMAGE_BOXES
)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    float* pitem = predict + output_cdim * position;

    // 直接筛选最大类别置信度（V8无objectness分支）
    float* class_confidence = pitem + 4;
    float  confidence = *class_confidence++;
    int    label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }

    // 类别置信度筛选
    if (confidence < confidence_threshold)
        return;

    // 原子操作计数有效检测框
    int index = atomicAdd(parray, 1);
    if (index >= MAX_IMAGE_BOXES)
        return;

    // 解码边界框并映射到原始图像尺寸
    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    // 写入输出（含预测框位置，用于V8 Seg掩码索引）
    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1=保留，0=忽略
    *pout_item++ = position;  // 预测框在模型输出中的位置
}

// 保留原IOU计算核函数，优化变量命名与代码缩进
static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom,
    float bleft, float btop, float bright, float bbottom
)
{
    // 计算交集坐标
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    // 计算交集面积（无交集则为0）
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    // 计算两个框的面积
    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);

    // 计算IOU（交集/并集）
    return c_area / (a_area + b_area - c_area);
}

// 保留原快速NMS核函数，优化代码块结构
static __global__ void fast_nms_kernel(
    float* bboxes, 
    int MAX_IMAGE_BOXES, 
    float threshold
)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    int count = min((int)*bboxes, MAX_IMAGE_BOXES);  // 有效检测框总数
    if (position >= count)
        return;

    // 指向当前检测框
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;

    // 遍历所有检测框，过滤重叠框
    for (int i = 0; i < count; ++i) {
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        // 跳过自身或不同类别的框
        if (i == position || pcurrent[5] != pitem[5])
            continue;

        // 对比置信度，过滤高重叠框
        if (pitem[4] >= pcurrent[4]) {
            // 置信度相等时，保留索引更小的框（避免重复过滤）
            if (pitem[4] == pcurrent[4] && i < position)
                continue;

            // 计算IOU，超过阈值则标记当前框为无效
            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0], pitem[1], pitem[2], pitem[3]
            );
            if (iou > threshold) {
                pcurrent[6] = 0;  // 0=忽略
                return;
            }
        }
    }
}

// 保留原网格维度计算函数，保持逻辑不变
static dim3 grid_dims(int numJobs)
{
    int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

// 保留原线程块维度计算函数，保持逻辑不变
static dim3 block_dims(int numJobs)
{
    return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

// 保留原解码调度器，优化条件分支格式
static void decode_kernel_invoker(
    float* predict, 
    int num_bboxes, 
    int num_classes,
    int output_cdim, 
    float confidence_threshold,
    float nms_threshold, 
    float* invert_affine_matrix,
    float* parray, 
    int MAX_IMAGE_BOXES,
    Type type, 
    cudaStream_t stream
)
{
    auto grid = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);

    // 根据YOLO类型选择对应解码核
    if (type == Type::V8 || type == Type::V8Seg) {
        checkKernel(decode_kernel_v8<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes, output_cdim,
            confidence_threshold, invert_affine_matrix, parray, MAX_IMAGE_BOXES
        ));
    } 
    else if (type == Type::V5Face) {
        checkKernel(decode_kernel_v5_face<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes, output_cdim,
            confidence_threshold, invert_affine_matrix, parray, MAX_IMAGE_BOXES
        ));
    } 
    else {  // 适配V3/V5/V7/X
        checkKernel(decode_kernel_common<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes, output_cdim,
            confidence_threshold, invert_affine_matrix, parray, MAX_IMAGE_BOXES
        ));
    }

    // 启动NMS核（过滤重叠框）
    grid = grid_dims(MAX_IMAGE_BOXES);
    block = block_dims(MAX_IMAGE_BOXES);
    checkKernel(fast_nms_kernel<<<grid, block, 0, stream>>>(
        parray, MAX_IMAGE_BOXES, nms_threshold
    ));
}

// 保留原V8 Seg掩码解码核函数，优化参数换行
static __global__ void decode_single_mask_kernel(
    int left, 
    int top, 
    float* mask_weights,
    float* mask_predict, 
    int mask_width, 
    int mask_height,
    unsigned char* mask_out, 
    int mask_dim, 
    int out_width,
    int out_height
)
{
    // 计算当前线程对应的掩码像素坐标
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= out_width || dy >= out_height)
        return;

    // 计算掩码像素在特征图上的对应坐标
    int sx = left + dx;
    int sy = top + dy;
    if (sx < 0 || sx >= mask_width || sy < 0 || sy >= mask_height) {
        mask_out[dy * out_width + dx] = 0;  // 超出范围设为背景
        return;
    }

    // 计算掩码权重与特征图像素的加权和
    float cumprod = 0;
    for (int ic = 0; ic < mask_dim; ++ic) {
        float cval = mask_predict[(ic * mask_height + sy) * mask_width + sx];
        float wval = mask_weights[ic];
        cumprod += cval * wval;
    }

    // Sigmoid激活并转为uint8（0=背景，255=目标）
    float alpha = 1.0f / (1.0f + exp(-cumprod));
    mask_out[dy * out_width + dx] = alpha * 255;
}

// 保留原掩码解码调度器，保持逻辑不变
static void decode_single_mask(
    float left, 
    float top, 
    float* mask_weights,
    float* mask_predict, 
    int mask_width, 
    int mask_height,
    unsigned char* mask_out, 
    int mask_dim, 
    int out_width,
    int out_height, 
    cudaStream_t stream
)
{
    // 2D线程布局（32x32块，适配掩码2D结构）
    dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
    dim3 block(32, 32);

    checkKernel(decode_single_mask_kernel<<<grid, block, 0, stream>>>(
        left, top, mask_weights, mask_predict, mask_width, mask_height,
        mask_out, mask_dim, out_width, out_height
    ));
}

// 保留原类型转字符串函数，优化switch格式
const char* type_name(Type type)
{
    switch (type) {
        case Type::V5:      return "YoloV5";
        case Type::V3:      return "YoloV3";
        case Type::V7:      return "YoloV7";
        case Type::X:       return "YoloX";
        case Type::V8:      return "YoloV8";
        case Type::V5Face:  return "YoloV5Face";
        default:            return "Unknow";
    }
}

// 保留原仿射矩阵结构体，优化代码换行
struct AffineMatrix {
    float i2d[6];  // 正向矩阵（原图→模型输入尺寸）
    float d2i[6];  // 反向矩阵（模型输入尺寸→原图）

    void compute(const std::tuple<int, int>& from, const std::tuple<int, int>& to)
    {
        // 计算缩放比（保持宽高比）
        float scale_x = get<0>(to) / (float)get<0>(from);
        float scale_y = get<1>(to) / (float)get<1>(from);
        float scale = std::min(scale_x, scale_y);

        // 计算正向矩阵（缩放+居中偏移）
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

        // 计算反向矩阵（2x3矩阵求逆）
        double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];  // 行列式
        D = D != 0. ? double(1.) / D : double(0.);
        double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
        double b1 = -A11 * i2d[2] - A12 * i2d[5];
        double b2 = -A21 * i2d[2] - A22 * i2d[5];

        // 赋值反向矩阵
        d2i[0] = A11;
        d2i[1] = A12;
        d2i[2] = b1;
        d2i[3] = A21;
        d2i[4] = A22;
        d2i[5] = b2;
    }
};

// 保留原实例分割掩码类，保持构造/析构逻辑不变
InstanceSegmentMap::InstanceSegmentMap(int width, int height)
{
    this->width = width;
    this->height = height;
    // 分配页锁定内存（加速CPU/GPU传输）
    checkRuntime(cudaMallocHost(&this->data, width * height));
}

InstanceSegmentMap::~InstanceSegmentMap()
{
    if (this->data) {
        checkRuntime(cudaFreeHost(this->data));  // 释放页锁定内存
        this->data = nullptr;
    }
    this->width = 0;
    this->height = 0;
}

// 保留原YOLO推理核心类，优化成员变量与函数格式
class InferImpl : public Infer<BoxArray> {
public:
    // 保留原成员变量，顺序不变
    shared_ptr<trt::Infer>                         trt_;
    string                                         engine_file_;
    Type                                           type_;
    float                                          confidence_threshold_;
    float                                          nms_threshold_;
    vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;
    trt::Memory<float> input_buffer_, bbox_predict_, output_boxarray_;
    trt::Memory<float> segment_predict_;
    int                network_input_width_, network_input_height_;
    Norm               normalize_;
    vector<int>        bbox_head_dims_;
    vector<int>        segment_head_dims_;
    int                num_classes_ = 0;
    bool               has_segment_ = false;
    bool               has_keyPoint = false;
    bool               isdynamic_model_ = false;
    vector<shared_ptr<trt::Memory<unsigned char>>> box_segment_cache_;

    // 保留原虚析构函数
    virtual ~InferImpl() = default;

    // 保留原内存调整函数，优化代码换行
    void adjust_memory(int batch_size)
    {
        // 计算单张输入图像元素总数（宽×高×3通道）
        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        input_buffer_.gpu(batch_size * input_numel);  // 分配输入GPU内存

        // 分配检测头输出GPU内存
        bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);

        // 分配检测框输出内存（含计数与框数据）
        output_boxarray_.gpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
        output_boxarray_.cpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));

        // 分割模型：分配分割头输出GPU内存
        if (has_segment_)
            segment_predict_.gpu(batch_size * segment_head_dims_[1] *
                                 segment_head_dims_[2] * segment_head_dims_[3]);

        // 调整预处理缓存数量（确保≥批次大小）
        if ((int)preprocess_buffers_.size() < batch_size) {
            for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
                preprocess_buffers_.push_back(make_shared<trt::Memory<unsigned char>>());
        }
    }

    // 保留原预处理函数，优化参数与代码块结构
    void preprocess(
        int ibatch, 
        const Image& image,
        shared_ptr<trt::Memory<unsigned char>> preprocess_buffer,
        AffineMatrix& affine, 
        void* stream = nullptr
    )
    {
        // 计算仿射矩阵（原图→模型输入尺寸）
        affine.compute(
            make_tuple(image.width, image.height),
            make_tuple(network_input_width_, network_input_height_)
        );

        // 计算内存大小
        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        float* input_device = input_buffer_.gpu() + ibatch * input_numel;  // 当前图像输入地址
        size_t size_image = image.width * image.height * 3;                // 图像字节数
        size_t size_matrix = upbound(sizeof(affine.d2i), 32);              // 仿射矩阵对齐后大小

        // 分配GPU工作空间（存储仿射矩阵+图像数据）
        uint8_t* gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);
        float*   affine_matrix_device = (float*)gpu_workspace;
        uint8_t* image_device = gpu_workspace + size_matrix;

        // 分配CPU工作空间（暂存数据）
        uint8_t* cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
        float*   affine_matrix_host = (float*)cpu_workspace;
        uint8_t* image_host = cpu_workspace + size_matrix;

        cudaStream_t stream_ = (cudaStream_t)stream;

        // CPU→CPU拷贝：图像数据与仿射矩阵
        std::chrono ::high_resolution_clock::time_point a1 = std::chrono::high_resolution_clock::now();
        memcpy(image_host, image.bgrptr, size_image);
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        std::chrono ::high_resolution_clock::time_point a2 = std::chrono::high_resolution_clock::now();
        auto time_used2 = std::chrono::duration_cast<std::chrono::duration<double>>(a2 - a1);

        // CPU→GPU异步拷贝（避免阻塞CPU）
        checkRuntime(cudaMemcpyAsync(
            image_device, image_host, size_image,
            cudaMemcpyHostToDevice, stream_
        ));
        checkRuntime(cudaMemcpyAsync(
            affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
            cudaMemcpyHostToDevice, stream_
        ));

        // CUDA核函数：双线性插值缩放 + 归一化（输出CHW格式float）
        warp_affine_bilinear_and_normalize_plane(
            image_device, image.width * 3,
            image.width, image.height,
            input_device, network_input_width_, network_input_height_,
            affine_matrix_device, 114,
            normalize_, stream_
        );
    }

    // 保留原引擎加载函数，优化条件分支格式
    bool load(
        const string& engine_file, 
        Type type,
        float confidence_threshold, 
        float nms_threshold
    )
    {
        // 加载TRT引擎
        trt_ = trt::load(engine_file);
        if (trt_ == nullptr)
            return false;
        trt_->print();  // 打印引擎信息

        // 初始化基础参数
        this->type_ = type;
        this->confidence_threshold_ = confidence_threshold;
        this->nms_threshold_ = nms_threshold;
        this->engine_file_ = engine_file;

        // 解析输入输出维度
        auto input_dim = trt_->static_dims(0);
        has_segment_ = (type == Type::V8Seg);   // 是否为分割模型
        has_keyPoint = (type == Type::V5Face);  // 是否含关键点

        // 分割模型：输出0=分割头，1=检测头；非分割模型：输出0=检测头
        if (has_segment_) {
            bbox_head_dims_ = trt_->static_dims(2);
            segment_head_dims_ = trt_->static_dims(1);
        } else {
            bbox_head_dims_ = trt_->static_dims(1);
        }

        // 初始化输入尺寸与动态批次标志
        network_input_width_ = input_dim[3];
        network_input_height_ = input_dim[2];
        isdynamic_model_ = trt_->has_dynamic_dim();

        // 配置归一化参数（按YOLO类型适配）
        if (type == Type::V5 || type == Type::V3 || type == Type::V7) {
            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 5;  // 类别数=通道数-5（cx,cy,w,h,objectness）
        } else if (type == Type::V8) {
            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 4;  // 类别数=通道数-4（无objectness）
        } else if (type == Type::V8Seg) {
            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 4 - segment_head_dims_[1];
        } else if (type == Type::X) {
            normalize_ = Norm::None();  // YoloX已提前归一化
            num_classes_ = bbox_head_dims_[2] - 5;
        } else if (type == Type::V5Face) {
            normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 5 - 2 * point_num;
        } else {
            INFO("Unsupport type %d", type);
            return false;
        }

        return true;
    }

    // 保留原单图推理函数
    virtual BoxArray forward(const Image& image, void* stream = nullptr) override
    {
        auto output = forwards({image}, stream);
        return output.empty() ? BoxArray() : output[0];
    }

    // 保留原批量推理函数，优化代码块拆分与缩进
    virtual vector<BoxArray> forwards(const vector<Image>& images, void* stream = nullptr) override
    {
        int num_image = images.size();
        if (num_image == 0)
            return {};

        // 1. 适配批次大小（动态/静态模型处理）
        auto input_dims = trt_->static_dims(0);
        int  infer_batch_size = input_dims[0];  // 初始批次大小

        if (infer_batch_size != num_image) {
            if (isdynamic_model_) {
                // 动态模型：调整批次大小
                infer_batch_size = num_image;
                input_dims[0] = num_image;
                if (!trt_->set_run_dims(0, input_dims))
                    return {};
            } else {
                // 静态模型：检查图像数≤最大批次
                if (infer_batch_size < num_image) {
                    INFO("When using static shape model, number of images[%d] must be "
                         "less than or equal to the maximum batch[%d].",
                         num_image, infer_batch_size);
                    return {};
                }
            }
        }

        // 2. 调整内存（输入/输出/缓存）
        std::chrono ::high_resolution_clock::time_point a1 = std::chrono::high_resolution_clock::now();
        adjust_memory(infer_batch_size);
        std::chrono ::high_resolution_clock::time_point a2 = std::chrono::high_resolution_clock::now();
        auto time_used2 = std::chrono::duration_cast<std::chrono::duration<double>>(a2 - a1);

        // 3. 批量预处理（每张图像计算仿射矩阵+缩放+归一化）
        vector<AffineMatrix> affine_matrixs(num_image);
        std::chrono ::high_resolution_clock::time_point a3 = std::chrono::high_resolution_clock::now();
        cudaStream_t stream_ = (cudaStream_t)stream;
        for (int i = 0; i < num_image; ++i) {
            preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);
        }

        // 4. 准备TensorRT前向绑定（输入+输出地址）
        float*        bbox_output_device = bbox_predict_.gpu();
        vector<void*> bindings{input_buffer_.gpu(), bbox_output_device};
        // 分割模型：绑定分割头输出
        if (has_segment_) {
            bindings = {input_buffer_.gpu(), segment_predict_.gpu(), bbox_output_device};
        }

        // 5. 执行TensorRT前向传播
        std::chrono ::high_resolution_clock::time_point a3d1 = std::chrono::high_resolution_clock::now();
        auto time_used3 = std::chrono::duration_cast<std::chrono::duration<double>>(a3d1 - a3);
        if (!trt_->forward(bindings, stream)) {
            INFO("Failed to tensorRT forward.");
            return {};
        }
        std::chrono ::high_resolution_clock::time_point a3d2 = std::chrono::high_resolution_clock::now();
        auto time_used3d2 = std::chrono::duration_cast<std::chrono::duration<double>>(a3d2 - a3d1);

        // 6. 批量解码+NMS（每张图像独立处理）
        for (int ib = 0; ib < num_image; ++ib) {
            // 指向当前图像的检测框输出地址（GPU）
            float* boxarray_device = output_boxarray_.gpu() +
                                    ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            // 指向当前图像的反向仿射矩阵（GPU）
            float* affine_matrix_device = (float*)preprocess_buffers_[ib]->gpu();
            // 指向当前图像的检测头输出（GPU）
            float* image_based_bbox_output = bbox_output_device +
                                            ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);

            // 初始化检测框计数为0
            checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));

            // 解码+NMS
            decode_kernel_invoker(
                image_based_bbox_output, bbox_head_dims_[1], num_classes_,
                bbox_head_dims_[2], confidence_threshold_, nms_threshold_,
                affine_matrix_device, boxarray_device, MAX_IMAGE_BOXES,
                type_, stream_
            );
        }

        // 7. GPU→CPU拷贝检测框结果
        std::chrono ::high_resolution_clock::time_point a3d3 = std::chrono::high_resolution_clock::now();
        auto time_used3d3 = std::chrono::duration_cast<std::chrono::duration<double>>(a3d3 - a3d2);
        checkRuntime(cudaMemcpyAsync(
            output_boxarray_.cpu(), output_boxarray_.gpu(),
            output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_
        ));
        checkRuntime(cudaStreamSynchronize(stream_));  // 等待拷贝完成
        std::chrono ::high_resolution_clock::time_point a4 = std::chrono::high_resolution_clock::now();
        auto time_used3d4 = std::chrono::duration_cast<std::chrono::duration<double>>(a4 - a3d3);
        auto time_used4 = std::chrono::duration_cast<std::chrono::duration<double>>(a4 - a3);
        INFO("forward and decode_kernel_invoker time: %f", time_used4.count() * 1000);

        // 8. 整理检测结果（含分割掩码生成）
        std::chrono ::high_resolution_clock::time_point a5 = std::chrono::high_resolution_clock::now();
        vector<BoxArray> arrout(num_image);
        int              imemory = 0;  // 分割掩码缓存索引

        for (int ib = 0; ib < num_image; ++ib) {
            // 指向当前图像的检测框结果（CPU）
            float* parray = output_boxarray_.cpu() +
                            ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            int count = min(MAX_IMAGE_BOXES, (int)*parray);  // 有效检测框数
            BoxArray& output = arrout[ib];
            output.reserve(count);  // 预分配内存

            // 遍历每个检测框
            for (int i = 0; i < count; ++i) {
                float* pbox = parray + 1 + i * NUM_BOX_ELEMENT;
                // 提取检测框参数
                float left = pbox[0], top = pbox[1], right = pbox[2], bottom = pbox[3];
                float confidence = pbox[4];
                int   label = pbox[5], keepflag = pbox[6];
                std::vector<float> points;  // 关键点（V5Face用）

                // 提取关键点
                for (int j = 1; j < 9; j++) {
                    points.push_back(pbox[6 + j]);
                }

                // 仅保留有效框（keepflag=1）
                if (keepflag == 1) {
                    Box result_box(left, top, right, bottom, confidence, label, points);

                    // 分割模型：生成实例掩码
                    if (has_segment_) {
                        int row_index = pbox[7];  // 预测框在检测头中的位置（掩码权重索引）
                        int mask_dim = segment_head_dims_[1];  // 掩码权重维度
                        // 指向当前框的掩码权重
                        float* mask_weights = bbox_output_device +
                                            (ib * bbox_head_dims_[1] + row_index) * bbox_head_dims_[2] +
                                            num_classes_ + 4;

                        // 计算掩码特征图上的框坐标（正向仿射矩阵映射）
                        float* i2d = affine_matrixs[ib].i2d;
                        float mask_left, mask_top, mask_right, mask_bottom;
                        affine_project(i2d, left, top, &mask_left, &mask_top);
                        affine_project(i2d, right, bottom, &mask_right, &mask_bottom);

                        // 计算掩码输出尺寸（匹配检测框大小）
                        float scale_to_predict_x = segment_head_dims_[3] / (float)network_input_width_;
                        float scale_to_predict_y = segment_head_dims_[2] / (float)network_input_height_;
                        int mask_out_width = (mask_right - mask_left) * scale_to_predict_x + 0.5f;
                        int mask_out_height = (mask_bottom - mask_top) * scale_to_predict_y + 0.5f;

                        if (mask_out_width > 0 && mask_out_height > 0) {
                            // 调整分割掩码缓存（复用内存）
                            if (imemory >= (int)box_segment_cache_.size()) {
                                box_segment_cache_.push_back(make_shared<trt::Memory<unsigned char>>());
                            }

                            // 分配掩码输出内存并解码
                            int bytes_of_mask_out = mask_out_width * mask_out_height;
                            auto box_segment_mem = box_segment_cache_[imemory++];
                            result_box.seg = make_shared<InstanceSegmentMap>(mask_out_width, mask_out_height);
                            unsigned char* mask_out_device = box_segment_mem->gpu(bytes_of_mask_out);
                            unsigned char* mask_out_host = result_box.seg->data;

                            // 生成掩码
                            decode_single_mask(
                                mask_left * scale_to_predict_x,
                                mask_top * scale_to_predict_y,
                                mask_weights,
                                segment_predict_.gpu() + ib * segment_head_dims_[1] *
                                    segment_head_dims_[2] * segment_head_dims_[3],
                                segment_head_dims_[3], segment_head_dims_[2],
                                mask_out_device, mask_dim,
                                mask_out_width, mask_out_height, stream_
                            );

                            // 掩码GPU→CPU拷贝
                            checkRuntime(cudaMemcpyAsync(
                                mask_out_host, mask_out_device, bytes_of_mask_out,
                                cudaMemcpyDeviceToHost, stream_
                            ));
                        }
                    }

                    output.emplace_back(result_box);  // 加入输出数组
                }
            }
        }

        // 等待分割掩码拷贝完成
        if (has_segment_)
            checkRuntime(cudaStreamSynchronize(stream_));

        std::chrono ::high_resolution_clock::time_point a6 = std::chrono::high_resolution_clock::now();
        auto time_used6 = std::chrono::duration_cast<std::chrono::duration<double>>(a6 - a5);
        return arrout;
    }
};

// 保留原原始指针创建函数，保持逻辑不变
Infer<BoxArray>* loadraw(
    const std::string& engine_file, 
    Type type,
    float confidence_threshold, 
    float nms_threshold
)
{
    InferImpl* impl = new InferImpl();
    if (!impl->load(engine_file, type, confidence_threshold, nms_threshold)) {
        delete impl;
        impl = nullptr;
    }
    return impl;
}

// 保留原智能指针创建函数，保持逻辑不变
shared_ptr<Infer<BoxArray>> load(
    const string& engine_file, 
    Type type,
    float confidence_threshold, 
    float nms_threshold
)
{
    return std::shared_ptr<InferImpl>((InferImpl*)loadraw(
        engine_file, type, confidence_threshold, nms_threshold
    ));
}

// 保留原HSV转BGR函数，优化switch格式
std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v)
{
    const int   h_i = static_cast<int>(h * 6);  // 色相分区（0~5）
    const float f = h * 6 - h_i;                 // 色相小数部分
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float       r, g, b;

    // 根据色相分区计算RGB值
    switch (h_i) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
        default: r = 1, g = 1, b = 1; break;
    }

    // RGB→BGR（适配OpenCV格式），转为uint8
    return make_tuple(
        static_cast<uint8_t>(b * 255),
        static_cast<uint8_t>(g * 255),
        static_cast<uint8_t>(r * 255)
    );
}

// 保留原随机颜色生成函数，保持逻辑不变
std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id)
{
    // 根据ID生成随机色相和饱和度（确保不同ID颜色不同）
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);  // 明度固定为1（亮色）
}

};  // namespace yolo
