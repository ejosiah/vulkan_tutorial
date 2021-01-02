//
// Created by Josiah on 12/28/2020.
//

#pragma once

#include <cstdint>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <cassert>
#include <vector>
#include <iostream>
#include <optional>
#include <set>
#include "vulkan_wrapper.h"
#include <glm/glm.hpp>
#include <array>

#define REPORT_ERROR(status, msg) if(status != VK_SUCCESS) throw std::runtime_error(msg);


constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;
constexpr VkAllocationCallbacks* NO_ALLOCATOR = nullptr;
constexpr uint64_t NO_TIMEOUT = UINT64_MAX;
using cstring_t = const char*;

struct Vertex{
    glm::vec2 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription(){
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescription(){
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

template<typename Func>
inline Func instanceProcAddress(VkInstance instance, const std::string& funcName){
    return reinterpret_cast<Func>(vkGetInstanceProcAddr(instance, funcName.c_str()));
}

inline VkResult CreateDebugUtilsMessengerEXT(
        VkInstance instance,
        const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
        const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger){

    auto func = instanceProcAddress<PFN_vkCreateDebugUtilsMessengerEXT>(instance, "vkCreateDebugUtilsMessengerEXT");
    if(func != nullptr){
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

}

inline void DestroyDebugUtilsMessengerEXT(
        VkInstance instance,
        VkDebugUtilsMessengerEXT pDebugMessenger,
        const VkAllocationCallbacks* pAllocator){

    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if(func != nullptr){
        func(instance, pDebugMessenger, pAllocator);
    }

}

struct QueueFamilyIndices{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    [[nodiscard]]
    constexpr bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }

    [[nodiscard]]
    std::set<uint32_t> uniqueQueueFamilies() const {
        return { graphicsFamily.value(), presentFamily.value()};
    }
};

struct SwapChainSupportedDetails{
    VkSurfaceCapabilitiesKHR  capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class Application {
public:

    void run();

private:
    void initWindow();

    void initVulkan();

    void mainLoop();

    void drawFrame();

    void createInstance();

    bool checkValidationLayerSupport();

    std::vector<cstring_t> getRequiredExtensions();

    void setupDebugMessenger();

    void pickPhysicalDevice();

    bool isDeviceSuitable(VkPhysicalDevice pDevice);

    void createLogicalDevice();

    void createSurface();

    bool checkDeviceExtensionSupport(VkPhysicalDevice pDevice);

    void createGraphicsPipeline();

    void createSwapChain();

    void recreateSwapChain();

    void cleanupSwapChain();

    void createImageViews();

    void createRenderPass();

    void createFramebuffers();

    void createCommandPool();

    void createVertexBuffer();

    void createCommandBuffers();

    void createSyncObjects();

    void cleanup();

    VkShaderModule createShaderModule(const std::vector<char>& code);

    std::vector<char> readFile(const std::string& filename);

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

    SwapChainSupportedDetails querySwapChainSupport(VkPhysicalDevice pDevice);

    VkDebugUtilsMessengerCreateInfoEXT&  populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData
            );

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

private:
    GLFWwindow * window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    VkRenderPass  renderPass;
    VkPipelineLayout pipelineLayout;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    // template<typename Type> Pipeline<Type>
    VkPipeline graphicsPipeline;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;

//    std::vector<VkFence> inFlightFences;
//    std::vector<VkFence> imagesInFlight;

    std::vector<vkn::Fence> inFlightFences;
    std::vector<vkn::Fence*> imagesInFlight;
    size_t currentFrame = 0;
    bool framebufferResized = false;

    const std::vector<const char*> validationLayers = {
            "VK_LAYER_KHRONOS_validation",
       //     "VK_LAYER_LUNARG_api_dump"
    };

    const std::vector<const char*> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    const std::vector<Vertex> vertices = {
            {{0.0f, -0.5f}, {1.0f, 1.0f, 0.0f}},
            {{0.5f, 0.5f}, {0.0f, 1.0f, 1.0f}},
            {{-0.5f, 0.5f}, {1.0f, 0.0f, 1.0f}}
    };

#ifdef NDEBUG
    const bool enabledValidationLayers = false
#else
    const bool enabledValidationLayers = true;
#endif
    VkDebugUtilsMessengerEXT debugMessenger;
};
