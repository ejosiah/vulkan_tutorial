#include <vulkan/vulkan.h>
#include <vector>
#include <algorithm>

namespace vkn{

    class Fence{
    private:
        Fence(VkFence vkFence, VkDevice device, VkAllocationCallbacks* allocator = nullptr);

    public:
        friend class FenceCreator;

        Fence();

        Fence(const Fence&) = delete;

        Fence(Fence&& srcFence) noexcept;

        Fence operator=(const Fence&) = delete;

        Fence& operator=(Fence&& srcFence) noexcept;

        ~Fence();

        void wait(uint64_t timeout = UINT64_MAX) const;

        void reset() const;

        inline explicit operator VkFence() const {
            return fence;
        }

        static void wait(const std::vector<Fence>& fences, bool waitForAll, uint64_t timeout = UINT64_MAX);

    protected:
        VkFence fence;
        VkDevice device;
        VkAllocationCallbacks* allocator;
    };

    class FenceCreator{
    public:
        FenceCreator(VkDevice device, VkAllocationCallbacks* allocator = nullptr);

        FenceCreator& signaled();

        Fence create();

    private:
        VkFenceCreateInfo info;
        VkDevice device;
        VkAllocationCallbacks* allocator;
    };

#include "detail/vk_fence_inl.h"
}