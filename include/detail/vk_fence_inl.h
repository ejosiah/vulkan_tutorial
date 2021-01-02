inline Fence::Fence(VkFence vkFence, VkDevice device, VkAllocationCallbacks* allocator)
    :fence{ vkFence }
    , device{ device }
    , allocator{ allocator }{

}

inline Fence::Fence(): fence{ nullptr}{}

inline Fence::Fence(Fence&& srcFence) noexcept{
    fence = srcFence.fence;
    device = srcFence.device;
    allocator = srcFence.allocator;

    srcFence.fence = nullptr;
}

inline Fence& Fence::operator=(Fence&& srcFence) noexcept{
    fence = srcFence.fence;
    device = srcFence.device;
    allocator = srcFence.allocator;

    srcFence.fence = nullptr;

    return *this;
}

inline Fence::~Fence(){
//    if(fence) {
//        vkDestroyFence(device, fence, allocator);
//    }
}

inline void Fence::wait(uint64_t timeout) const {
    vkWaitForFences(device, 1, &fence, VK_TRUE, timeout);
}

inline void Fence::reset() const {
    vkResetFences(device, 1, &fence);
}

inline void Fence::wait(const std::vector<Fence>& fences, bool waitForAll, uint64_t timeout){
    std::vector<VkFence> vkFences(fences.size());
    std::transform(begin(fences), end(fences), begin(vkFences), [](auto& it){ return static_cast<VkFence>(it); });
    auto fenceCount = static_cast<uint32_t>(fences.size());
    vkWaitForFences(fences.front().device, fenceCount, vkFences.data(), VK_TRUE, timeout);
}

inline FenceCreator::FenceCreator(VkDevice device, VkAllocationCallbacks* allocator)
    : info{}
    , device{device}
    , allocator{allocator}
{
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
}

inline FenceCreator& FenceCreator::signaled(){
    info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    return *this;
}

inline Fence FenceCreator::create(){
    VkFence fence;
    vkCreateFence(device, &info, allocator, &fence);
    return Fence(fence, device, allocator);
}