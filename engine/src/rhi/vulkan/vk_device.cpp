// Vulkan device creation and resource objects. Command recording is in vk_command.cpp.
#include <cstdio>
#include <cstring>
#include <fstream>

#include "rhi/vulkan/vk_internal.h"

namespace engine {
namespace rhi {

std::unique_ptr<Device> createDevice(Backend backend, const DeviceOptions& options) {
    if (backend != Backend::Vulkan) throw Error("only the Vulkan backend exists so far");
    return std::make_unique<vk::DeviceImpl>(options);
}

namespace vk {

namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
    auto* ctx = static_cast<Context*>(user);
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ctx->validationErrors++;
    if (severity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT))
        std::fprintf(stderr, "[vulkan validation] %s\n", data->pMessage);
    return VK_FALSE;
}

bool hasInstanceLayer(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> props(count);
    vkEnumerateInstanceLayerProperties(&count, props.data());
    for (const auto& p : props)
        if (std::strcmp(p.layerName, name) == 0) return true;
    return false;
}

std::vector<uint32_t> readSpirv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw Error("cannot open SPIR-V file " + path);
    const std::streamsize size = f.tellg();
    if (size <= 0 || size % 4 != 0) throw Error("bad SPIR-V file " + path);
    std::vector<uint32_t> words(static_cast<size_t>(size) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(words.data()), size);
    return words;
}

VkFormat vertexFormat(VertexFormat f) {
    switch (f) {
        case VertexFormat::Float1: return VK_FORMAT_R32_SFLOAT;
        case VertexFormat::Float2: return VK_FORMAT_R32G32_SFLOAT;
        case VertexFormat::Float3: return VK_FORMAT_R32G32B32_SFLOAT;
        case VertexFormat::Float4: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    return VK_FORMAT_UNDEFINED;
}

}  // namespace

VkFormat toVkFormat(Format f) {
    switch (f) {
        case Format::RGBA8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        case Format::RGBA16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case Format::R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
        case Format::D32_FLOAT: return VK_FORMAT_D32_SFLOAT;
    }
    return VK_FORMAT_UNDEFINED;
}

VkImageAspectFlags aspectOf(Format f) {
    return f == Format::D32_FLOAT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
}

VkImageLayout layoutFor(ImageState s) {
    switch (s) {
        case ImageState::Undefined: return VK_IMAGE_LAYOUT_UNDEFINED;
        case ImageState::ColorTarget: return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        case ImageState::DepthTarget: return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        case ImageState::ShaderRead: return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        case ImageState::TransferSrc: return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        case ImageState::TransferDst: return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    }
    return VK_IMAGE_LAYOUT_UNDEFINED;
}

uint32_t Context::findMemory(uint32_t typeBits, VkMemoryPropertyFlags props) const {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) return i;
    throw Error("no suitable Vulkan memory type");
}

// ---------- DeviceImpl ----------

DeviceImpl::DeviceImpl(const DeviceOptions& options) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "engine";
    app.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> layers, exts;
    if (options.validation) {
        if (!hasInstanceLayer(kValidationLayer)) throw Error("validation requested but the Khronos layer is not installed");
        layers.push_back(kValidationLayer);
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = static_cast<uint32_t>(layers.size());
    ici.ppEnabledLayerNames = layers.data();
    ici.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    ici.ppEnabledExtensionNames = exts.data();
    VK_CHECK(vkCreateInstance(&ici, nullptr, &ctx.instance), "vkCreateInstance");

    if (options.validation) {
        auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(ctx.instance, "vkCreateDebugUtilsMessengerEXT"));
        if (!create) throw Error("vkCreateDebugUtilsMessengerEXT not found");
        VkDebugUtilsMessengerCreateInfoEXT mci{};
        mci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        mci.pfnUserCallback = debugCallback;
        mci.pUserData = &ctx;
        VK_CHECK(create(ctx.instance, &mci, nullptr, &ctx.messenger), "vkCreateDebugUtilsMessengerEXT");
    }

    // Physical device: first one with a graphics queue and dynamic rendering (Vulkan 1.3 core).
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(ctx.instance, &count, devices.data());
    bool found = false;
    for (VkPhysicalDevice d : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(d, &props);
        if (props.apiVersion < VK_API_VERSION_1_3) continue;
        VkPhysicalDeviceVulkan13Features f13{};
        f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &f13;
        vkGetPhysicalDeviceFeatures2(d, &f2);
        if (!f13.dynamicRendering) continue;
        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qCount, qprops.data());
        for (uint32_t i = 0; i < qCount; i++) {
            if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                ctx.phys = d;
                ctx.queueFamily = i;
                name_ = props.deviceName;
                found = true;
                break;
            }
        }
        if (found) break;
    }
    if (!found) throw Error("no Vulkan 1.3 device with dynamic rendering and a graphics queue");
    vkGetPhysicalDeviceMemoryProperties(ctx.phys, &ctx.memProps);

    VkPhysicalDeviceVulkan13Features enable13{};
    enable13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    enable13.dynamicRendering = VK_TRUE;

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = ctx.queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &enable13;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VK_CHECK(vkCreateDevice(ctx.phys, &dci, nullptr, &ctx.device), "vkCreateDevice");
    vkGetDeviceQueue(ctx.device, ctx.queueFamily, 0, &ctx.queue);

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = ctx.queueFamily;
    VK_CHECK(vkCreateCommandPool(ctx.device, &cpci, nullptr, &ctx.cmdPool), "vkCreateCommandPool");
    // GPU timestamps for per-pass timing (see CommandList::writeTimestamp).
    VkPhysicalDeviceProperties devProps{};
    vkGetPhysicalDeviceProperties(ctx.phys, &devProps);
    ctx.timestamps = devProps.limits.timestampComputeAndGraphics == VK_TRUE;
    ctx.timestampPeriod = devProps.limits.timestampPeriod;
    if (ctx.timestamps) {
        VkQueryPoolCreateInfo qpci{};
        qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = kMaxTimestamps;
        VK_CHECK(vkCreateQueryPool(ctx.device, &qpci, nullptr, &ctx.timestampPool), "timestamp query pool");
    }

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(ctx.device, &fci, nullptr, &ctx.fence), "vkCreateFence");

    // Fixed binding layout: uniform buffer at 0, sampled images at bindings 1..kMaxSampledImages.
    VkDescriptorSetLayoutBinding bindings[1 + kMaxSampledImages + kMaxStorageBuffers]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    for (uint32_t i = 0; i < kMaxSampledImages; i++) {
        bindings[1 + i].binding = 1 + i;
        bindings[1 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1 + i].descriptorCount = 1;
        bindings[1 + i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    // Storage buffers: compute passes use slots 0 and 1 as in and out; the ocean vertex stage reads its fields.
    for (uint32_t slot = 0; slot < kMaxStorageBuffers; slot++) {
        VkDescriptorSetLayoutBinding& b = bindings[1 + kMaxSampledImages + slot];
        b.binding = 1 + kMaxSampledImages + slot;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo slci{};
    slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    slci.bindingCount = 1 + kMaxSampledImages + kMaxStorageBuffers;
    slci.pBindings = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &slci, nullptr, &ctx.setLayout), "descriptor set layout");

    VkDescriptorPoolSize sizes[3]{};
    sizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 512};
    sizes[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2048};
    sizes[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024};
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 256;
    dpci.poolSizeCount = 3;
    dpci.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &ctx.descPool), "descriptor pool");

    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    // Trilinear: images with one level clamp to level 0, so render targets are unaffected.
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.minLod = 0.0f;
    sci.maxLod = VK_LOD_CLAMP_NONE;
    VK_CHECK(vkCreateSampler(ctx.device, &sci, nullptr, &ctx.sampler), "sampler");
}

DeviceImpl::~DeviceImpl() {
    if (ctx.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(ctx.device);
        vkDestroySampler(ctx.device, ctx.sampler, nullptr);
        vkDestroyDescriptorPool(ctx.device, ctx.descPool, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device, ctx.setLayout, nullptr);
        vkDestroyFence(ctx.device, ctx.fence, nullptr);
        vkDestroyQueryPool(ctx.device, ctx.timestampPool, nullptr);
        vkDestroyCommandPool(ctx.device, ctx.cmdPool, nullptr);
        vkDestroyDevice(ctx.device, nullptr);
    }
    if (ctx.messenger != VK_NULL_HANDLE) {
        auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(ctx.instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy) destroy(ctx.instance, ctx.messenger, nullptr);
    }
    if (ctx.instance != VK_NULL_HANDLE) vkDestroyInstance(ctx.instance, nullptr);
}

std::unique_ptr<Image> DeviceImpl::createImage(const ImageDesc& desc) {
    return std::make_unique<ImageImpl>(ctx, desc);
}
std::unique_ptr<Buffer> DeviceImpl::createBuffer(size_t bytes, BufferUsage usage) {
    return std::make_unique<BufferImpl>(ctx, bytes, usage);
}
std::unique_ptr<Shader> DeviceImpl::createShader(const std::string& spirvPath) {
    return std::make_unique<ShaderImpl>(ctx, spirvPath);
}
std::unique_ptr<Pipeline> DeviceImpl::createPipeline(const PipelineDesc& desc) {
    return std::make_unique<PipelineImpl>(ctx, desc);
}
std::unique_ptr<Pipeline> DeviceImpl::createComputePipeline(const ComputePipelineDesc& desc) {
    return std::make_unique<PipelineImpl>(ctx, desc);
}
void DeviceImpl::setStorageBuffer(DescriptorSet& setBase, Buffer& buffer, uint32_t slot) {
    if (slot >= kMaxStorageBuffers) throw Error("storage buffer slot out of range");
    auto& set = static_cast<DescriptorSetImpl&>(setBase);
    auto* sb = static_cast<BufferImpl*>(&buffer);
    VkDescriptorBufferInfo info{};
    info.buffer = sb->buffer;
    info.offset = 0;
    info.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = set.set;
    w.dstBinding = 1 + kMaxSampledImages + slot;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &info;
    vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
}
std::unique_ptr<DescriptorSet> DeviceImpl::createDescriptorSet() {
    return std::make_unique<DescriptorSetImpl>(ctx);
}
std::unique_ptr<CommandList> DeviceImpl::createCommandList() {
    return std::make_unique<CommandListImpl>(ctx);
}

void DeviceImpl::updateDescriptorSet(DescriptorSet& setBase, const Buffer* uniform,
                                     const std::array<const Image*, kMaxSampledImages>& textures) {
    auto& set = static_cast<DescriptorSetImpl&>(setBase);
    VkWriteDescriptorSet writes[1 + kMaxSampledImages]{};
    VkDescriptorBufferInfo bufInfo{};
    VkDescriptorImageInfo imgInfos[kMaxSampledImages]{};
    uint32_t n = 0;
    if (uniform) {
        auto* ub = static_cast<const BufferImpl*>(uniform);
        bufInfo.buffer = ub->buffer;
        bufInfo.offset = 0;
        bufInfo.range = VK_WHOLE_SIZE;
        writes[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[n].dstSet = set.set;
        writes[n].dstBinding = 0;
        writes[n].descriptorCount = 1;
        writes[n].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[n].pBufferInfo = &bufInfo;
        n++;
    }
    for (uint32_t i = 0; i < kMaxSampledImages; i++) {
        if (!textures[i]) continue;
        auto* img = static_cast<const ImageImpl*>(textures[i]);
        imgInfos[i].sampler = ctx.sampler;
        imgInfos[i].imageView = img->view;
        imgInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        writes[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[n].dstSet = set.set;
        writes[n].dstBinding = 1 + i;
        writes[n].descriptorCount = 1;
        writes[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[n].pImageInfo = &imgInfos[i];
        n++;
    }
    if (n) vkUpdateDescriptorSets(ctx.device, n, writes, 0, nullptr);
}

std::vector<uint64_t> DeviceImpl::readTimestamps(uint32_t count) {
    if (!ctx.timestamps || count == 0) return {};
    if (count > kMaxTimestamps) throw Error("too many timestamps requested");
    std::vector<uint64_t> ticks(count);
    VK_CHECK(vkGetQueryPoolResults(ctx.device, ctx.timestampPool, 0, count, count * sizeof(uint64_t), ticks.data(),
                                   sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
             "vkGetQueryPoolResults");
    return ticks;
}

double DeviceImpl::timestampPeriodNs() const { return ctx.timestampPeriod; }

bool DeviceImpl::timestampsSupported() const { return ctx.timestamps; }

void DeviceImpl::submitAndWait(CommandList& listBase) {
    auto& list = static_cast<CommandListImpl&>(listBase);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &list.cb;
    VK_CHECK(vkResetFences(ctx.device, 1, &ctx.fence), "vkResetFences");
    VK_CHECK(vkQueueSubmit(ctx.queue, 1, &si, ctx.fence), "vkQueueSubmit");
    VK_CHECK(vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
}

// ---------- ImageImpl ----------

ImageImpl::ImageImpl(Context& c, const ImageDesc& desc)
    : ctx(c), width_(desc.width), height_(desc.height), format_(desc.format), name_(desc.name) {
    if (width_ == 0 || height_ == 0) throw Error("image '" + name_ + "' has zero size");
    VkImageUsageFlags usage = 0;
    if (desc.colorTarget) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (desc.depthTarget) usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (desc.sampled) usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (desc.transferSrc) usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;  // every image may be uploaded into
    if (usage == 0) throw Error("image '" + name_ + "' has no usage");
    aspect = aspectOf(format_);
    mipLevels = desc.mipLevels == 0 ? 1 : desc.mipLevels;

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = toVkFormat(format_);
    ici.extent = {width_, height_, 1};
    ici.mipLevels = mipLevels;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(ctx.device, &ici, nullptr, &image), "vkCreateImage");

    VkMemoryRequirements mr{};
    vkGetImageMemoryRequirements(ctx.device, image, &mr);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = ctx.findMemory(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(ctx.device, &mai, nullptr, &memory), "vkAllocateMemory(image)");
    VK_CHECK(vkBindImageMemory(ctx.device, image, memory, 0), "vkBindImageMemory");

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = toVkFormat(format_);
    vci.subresourceRange = {aspect, 0, mipLevels, 0, 1};
    VK_CHECK(vkCreateImageView(ctx.device, &vci, nullptr, &view), "vkCreateImageView");
}

ImageImpl::~ImageImpl() {
    vkDestroyImageView(ctx.device, view, nullptr);
    vkDestroyImage(ctx.device, image, nullptr);
    vkFreeMemory(ctx.device, memory, nullptr);
}

// ---------- BufferImpl ----------

BufferImpl::BufferImpl(Context& c, size_t bytes, BufferUsage usage) : ctx(c), size_(bytes) {
    if (bytes == 0) throw Error("zero-sized buffer");
    VkBufferUsageFlags flags = 0;
    switch (usage) {
        case BufferUsage::Vertex: flags = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; break;
        case BufferUsage::Index: flags = VK_BUFFER_USAGE_INDEX_BUFFER_BIT; break;
        case BufferUsage::Uniform: flags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT; break;
        case BufferUsage::Readback: flags = VK_BUFFER_USAGE_TRANSFER_DST_BIT; break;
        case BufferUsage::Staging: flags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT; break;
        case BufferUsage::Storage: flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; break;
    }
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = flags;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(ctx.device, &bci, nullptr, &buffer), "vkCreateBuffer");

    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(ctx.device, buffer, &mr);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    // Host-visible and coherent for every buffer: simple and correct first. Device-local
    // geometry is a later optimisation.
    mai.memoryTypeIndex = ctx.findMemory(mr.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(ctx.device, &mai, nullptr, &memory), "vkAllocateMemory(buffer)");
    VK_CHECK(vkBindBufferMemory(ctx.device, buffer, memory, 0), "vkBindBufferMemory");
    VK_CHECK(vkMapMemory(ctx.device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");
}

BufferImpl::~BufferImpl() {
    vkUnmapMemory(ctx.device, memory);
    vkDestroyBuffer(ctx.device, buffer, nullptr);
    vkFreeMemory(ctx.device, memory, nullptr);
}

// ---------- ShaderImpl ----------

ShaderImpl::ShaderImpl(Context& c, const std::string& spirvPath) : ctx(c) {
    const std::vector<uint32_t> words = readSpirv(spirvPath);
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = words.size() * sizeof(uint32_t);
    smci.pCode = words.data();
    VK_CHECK(vkCreateShaderModule(ctx.device, &smci, nullptr, &module), "vkCreateShaderModule " + spirvPath);
}

ShaderImpl::~ShaderImpl() { vkDestroyShaderModule(ctx.device, module, nullptr); }

// ---------- PipelineImpl ----------

PipelineImpl::PipelineImpl(Context& c, const ComputePipelineDesc& desc)
    : ctx(c), pushBytes(desc.pushConstantBytes), compute(true) {
    if (!desc.compute) throw Error("compute pipeline without a shader");
    if (desc.pushConstantBytes > 128) throw Error("push constants above the 128-byte minimum");
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = desc.pushConstantBytes;
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &ctx.setLayout;
    plci.pushConstantRangeCount = desc.pushConstantBytes ? 1 : 0;
    plci.pPushConstantRanges = desc.pushConstantBytes ? &range : nullptr;
    VK_CHECK(vkCreatePipelineLayout(ctx.device, &plci, nullptr, &layout), "vkCreatePipelineLayout (compute)");
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = static_cast<const ShaderImpl*>(desc.compute)->module;
    cpci.stage.pName = "main";
    cpci.layout = layout;
    VK_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline),
             "vkCreateComputePipelines");
}

PipelineImpl::PipelineImpl(Context& c, const PipelineDesc& desc) : ctx(c), pushBytes(desc.pushConstantBytes), hasColor2(desc.hasColor2), colorFormat2(desc.colorFormat2) {
    if (!desc.vertex || !desc.fragment) throw Error("pipeline '" + desc.name + "' needs both shader stages");
    if (desc.pushConstantBytes > 128) throw Error("push constants above the 128-byte minimum");
    const auto* vs = static_cast<const ShaderImpl*>(desc.vertex);
    const auto* fs = static_cast<const ShaderImpl*>(desc.fragment);

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    range.size = desc.pushConstantBytes;
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &ctx.setLayout;
    plci.pushConstantRangeCount = desc.pushConstantBytes ? 1 : 0;
    plci.pPushConstantRanges = desc.pushConstantBytes ? &range : nullptr;
    VK_CHECK(vkCreatePipelineLayout(ctx.device, &plci, nullptr, &layout), "vkCreatePipelineLayout");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs->module;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs->module;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    std::vector<VkVertexInputAttributeDescription> attrs;
    VkPipelineVertexInputStateCreateInfo vis{};
    vis.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (desc.vertexStride > 0) {
        binding.binding = 0;
        binding.stride = desc.vertexStride;
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        for (const auto& a : desc.vertexAttributes)
            attrs.push_back({a.location, 0, vertexFormat(a.format), a.offset});
        vis.vertexBindingDescriptionCount = 1;
        vis.pVertexBindingDescriptions = &binding;
        vis.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
        vis.pVertexAttributeDescriptions = attrs.data();
    }

    VkPipelineInputAssemblyStateCreateInfo ias{};
    ias.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vps{};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = desc.cullBack ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = desc.hasDepth && desc.depthTest ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = desc.hasDepth && desc.depthWrite ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                         VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendAttachmentState cbas[2]{cba, cba};
    VkPipelineColorBlendStateCreateInfo cbs{};
    cbs.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbs.attachmentCount = desc.hasColor2 ? 2 : 1;
    cbs.pAttachments = desc.hasColor2 ? cbas : &cba;

    const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dsi{};
    dsi.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dsi.dynamicStateCount = 2;
    dsi.pDynamicStates = dyn;

    // Dynamic rendering: the attachment formats are declared here instead of a render pass.
    VkFormat colorFmts[2]{toVkFormat(desc.colorFormat), toVkFormat(desc.colorFormat2)};
    VkPipelineRenderingCreateInfo prci{};
    prci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prci.colorAttachmentCount = desc.hasColor2 ? 2 : 1;
    prci.pColorAttachmentFormats = colorFmts;
    prci.depthAttachmentFormat = desc.hasDepth ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_UNDEFINED;

    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.pNext = &prci;
    gpci.stageCount = 2;
    gpci.pStages = stages;
    gpci.pVertexInputState = &vis;
    gpci.pInputAssemblyState = &ias;
    gpci.pViewportState = &vps;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms;
    gpci.pDepthStencilState = &ds;
    gpci.pColorBlendState = &cbs;
    gpci.pDynamicState = &dsi;
    gpci.layout = layout;
    gpci.renderPass = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline),
             "vkCreateGraphicsPipelines '" + desc.name + "'");
}

PipelineImpl::~PipelineImpl() {
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, layout, nullptr);
}

// ---------- DescriptorSetImpl ----------

DescriptorSetImpl::DescriptorSetImpl(Context& c) : ctx(c) {
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = ctx.descPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &ctx.setLayout;
    VK_CHECK(vkAllocateDescriptorSets(ctx.device, &dsai, &set), "vkAllocateDescriptorSets");
}

DescriptorSetImpl::~DescriptorSetImpl() { vkFreeDescriptorSets(ctx.device, ctx.descPool, 1, &set); }

}  // namespace vk
}  // namespace rhi
}  // namespace engine
