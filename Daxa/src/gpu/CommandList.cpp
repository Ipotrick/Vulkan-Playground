#include "CommandList.hpp"

namespace daxa {
	namespace gpu {
		CommandList::CommandList() {
			printf("command list creating\n");
			this->renderAttachmentBuffer.reserve(5);
			usedImages.reserve(10);
			usedBuffers.reserve(10);
			usedSets.reserve(10);
			usedGraphicsPipelines.reserve(3);
			usedStagingBuffers.reserve(1);
		}

		CommandList::~CommandList() {
			if (device) {
				printf("command list destruction\n");
				DAXA_ASSERT_M(operationsInProgress == 0, "a command list can not be descroyed when there are still commands recorded");
				DAXA_ASSERT_M(empty, "a command list can not be destroyed when not empty");
				vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
				vkDestroyCommandPool(device, cmdPool, nullptr);
				std::memset(this, 0, sizeof(CommandList));
			}
		}

		void CommandList::begin() {
			DAXA_ASSERT_M(usesOnGPU == 0, "can not change command list, that is currently used on gpu");
			operationsInProgress += 1;
			VkCommandBufferBeginInfo cbbi{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				.pNext = nullptr,
				.flags = VkCommandBufferUsageFlagBits::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
			};
			vkBeginCommandBuffer(cmd, &cbbi);
		}

		void CommandList::end() {
			DAXA_ASSERT_M(usesOnGPU == 0, "can not change command list, that is currently used on gpu");
			operationsInProgress -= 1;
			vkEndCommandBuffer(cmd);
		}

		void CommandList::copyHostToBuffer(HostToBufferCopyInfo copyInfo) {
			DAXA_ASSERT_M(usesOnGPU == 0, "can not change command list, that is currently used on gpu");
			if (copyInfo.size > STAGING_BUFFER_POOL_BUFFER_SIZE) {
				DAXA_ASSERT_M(false, "Currently uploads over a size of 67.108.864 bytes are not supported by the uploadToBuffer function. Please use a staging buffer.");
				usedBuffers.push_back(copyInfo.dst);
			}
			else {
				if (usedStagingBuffers.empty() || usedStagingBuffers.back().getLeftOverSize() < copyInfo.size) {
					usedStagingBuffers.push_back(stagingBufferPool.lock()->getStagingBuffer());
				}

				auto& stagingBuffer = usedStagingBuffers.back();

				auto offset = stagingBuffer.usedUpSize;

				stagingBuffer.buffer->upload(copyInfo.src, copyInfo.size, offset);

				stagingBuffer.usedUpSize += copyInfo.size;

				BufferToBufferCopyInfo btbCopyInfo{
					.src = stagingBuffer.buffer,
					.dst = copyInfo.dst,
					.region = BufferCopyRegion{
						.srcOffset = offset,
						.dstOffset = copyInfo.dstOffset,
						.size = copyInfo.size,
					}
				};
				copyBufferToBuffer(btbCopyInfo);
			}
		}

		void CommandList::copyHostToImage(HostToImageCopyInfo copyInfo) {
			DAXA_ASSERT_M(usesOnGPU == 0, "can not change command list, that is currently used on gpu");
			if (copyInfo.size > STAGING_BUFFER_POOL_BUFFER_SIZE) {
				DAXA_ASSERT_M(false, "Currently uploads over a size of 67.108.864 bytes are not supported by the uploadToBuffer function. Please use a staging buffer.");
				usedImages.push_back(copyInfo.dst);
			}
			else {
				if (usedStagingBuffers.empty() || usedStagingBuffers.back().getLeftOverSize() < copyInfo.size) {
					usedStagingBuffers.push_back(stagingBufferPool.lock()->getStagingBuffer());
				}

				auto& stagingBuffer = usedStagingBuffers.back();

				auto offset = stagingBuffer.usedUpSize;

				stagingBuffer.buffer->upload(copyInfo.src, copyInfo.size, offset);

				stagingBuffer.usedUpSize += copyInfo.size;

				BufferToImageCopyInfo btiCopy{
					.src = stagingBuffer.buffer,
					.dst = copyInfo.dst,
					.srcOffset = offset,
					.subRessourceLayers = copyInfo.dstImgSubressource,
					.size = copyInfo.size,
				};
				copyBufferToImage(btiCopy);
			}
		}

		void CommandList::copyMultiBufferToBuffer(BufferToBufferMultiCopyInfo copyInfo) {
			DAXA_ASSERT_M(usesOnGPU == 0, "can not change command list, that is currently used on gpu");
			DAXA_ASSERT_M(copyInfo.regions.size() > 0, "ERROR: tried copying 0 regions from buffer to buffer, this is a bug!");
			for (int i = 0; i < copyInfo.regions.size(); i++) {
				DAXA_ASSERT_M(copyInfo.src->getSize() >= copyInfo.regions[i].size + copyInfo.regions[i].srcOffset, "ERROR: src buffer is smaller than the region that shouly be copied!");
				DAXA_ASSERT_M(copyInfo.dst->getSize() >= copyInfo.regions[i].size + copyInfo.regions[i].dstOffset, "ERROR: dst buffer is smaller than the region that shouly be copied!");
			}
			usedBuffers.push_back(copyInfo.src);
			usedBuffers.push_back(copyInfo.dst);
			vkCmdCopyBuffer(cmd, copyInfo.src->getVkBuffer(), copyInfo.dst->getVkBuffer(), copyInfo.regions.size(), (VkBufferCopy*)copyInfo.regions.data());	// THIS COULD BREAK ON ABI CHANGE
		}

		void CommandList::copyBufferToBuffer(BufferToBufferCopyInfo copyInfo) {
			BufferToBufferMultiCopyInfo btbMultiCopy{
				.src = copyInfo.src,
				.dst = copyInfo.dst,
				.regions = { &copyInfo.region, 1 }
			};
			copyMultiBufferToBuffer(btbMultiCopy);
		}

		void CommandList::copyBufferToImage(BufferToImageCopyInfo copyInfo) {
			usedBuffers.push_back(copyInfo.src);
			usedImages.push_back(copyInfo.dst);

			VkImageSubresourceLayers imgSubRessource = copyInfo.subRessourceLayers.value_or(VkImageSubresourceLayers{
				.aspectMask = copyInfo.dst->getVkAspect(),
				.mipLevel = 0,
				.baseArrayLayer = 0,
				.layerCount = copyInfo.dst->getVkArrayLayers(),
				});

			VkBufferImageCopy bufferImageCopy{
				.bufferOffset = copyInfo.srcOffset,
				.bufferRowLength = 0,
				.bufferImageHeight = 0,
				.imageSubresource = imgSubRessource,
				.imageOffset = 0,
				.imageExtent = copyInfo.dst->getVkExtent(),
			};

			vkCmdCopyBufferToImage(cmd, copyInfo.src->getVkBuffer(), copyInfo.dst->getVkImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bufferImageCopy);
		}

		void CommandList::bindVertexBuffer(u32 binding, BufferHandle buffer, size_t bufferOffset) {
			DAXA_ASSERT_M(usesOnGPU == 0, "can not change command list, that is currently used on gpu");
			auto vkBuffer = buffer->getVkBuffer();
			vkCmdBindVertexBuffers(cmd, binding, 1, &vkBuffer, &bufferOffset);
		}

		void CommandList::beginRendering(BeginRenderingInfo ri) {
			DAXA_ASSERT_M(usesOnGPU == 0, "can not change command list, that is currently used on gpu");
			operationsInProgress += 1;
			for (int i = 0; i < ri.colorAttachments.size(); i++) {
				usedImages.push_back(ri.colorAttachments[i].image);

				renderAttachmentBuffer.push_back(VkRenderingAttachmentInfoKHR{
					.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
					.pNext = nullptr,
					.imageView = ri.colorAttachments[i].image->getVkView(),
					.imageLayout = ri.colorAttachments[i].layout,
					.resolveMode = VK_RESOLVE_MODE_NONE,// ri.colorAttachments[i].resolveMode,
					.loadOp = ri.colorAttachments[i].loadOp,
					.storeOp = ri.colorAttachments[i].storeOp,
					.clearValue = ri.colorAttachments[i].clearValue
				});
			}

			std::optional<VkRenderingAttachmentInfoKHR> depthAttachmentInfo = ri.depthAttachment == nullptr ? std::nullopt :
				std::optional{
					VkRenderingAttachmentInfoKHR{
						.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
						.pNext = nullptr,
						.imageView = ri.depthAttachment->image->getVkView(),
						.imageLayout = ri.depthAttachment->layout,
						.resolveMode = ri.depthAttachment->resolveMode,
						.loadOp = ri.depthAttachment->loadOp,
						.storeOp = ri.depthAttachment->storeOp
					}
				};

			std::optional<VkRenderingAttachmentInfoKHR> stencilAttachmentInfo = ri.stencilAttachment == nullptr ? std::nullopt :
				std::optional{
					VkRenderingAttachmentInfoKHR{
						.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
						.pNext = nullptr,
						.imageView = ri.stencilAttachment->image->getVkView(),
						.imageLayout = ri.stencilAttachment->layout,
						.resolveMode = ri.stencilAttachment->resolveMode,
						.loadOp = ri.stencilAttachment->loadOp,
						.storeOp = ri.stencilAttachment->storeOp
					}
				};

			VkRenderingInfoKHR renderInfo{};
			renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
			renderInfo.pNext = nullptr;
			if (ri.renderArea) {
				renderInfo.renderArea = *ri.renderArea;
			}
			else if (ri.colorAttachments.size() > 0) {
				renderInfo.renderArea.extent.width = ri.colorAttachments[0].image->getVkExtent().width;
				renderInfo.renderArea.extent.height = ri.colorAttachments[0].image->getVkExtent().height;
			}
			else if (ri.depthAttachment != nullptr) {
				renderInfo.renderArea.extent.width = ri.depthAttachment->image->getVkExtent().width;
				renderInfo.renderArea.extent.height = ri.depthAttachment->image->getVkExtent().height;
			}
			else if (ri.stencilAttachment != nullptr) {
				renderInfo.renderArea.extent.width = ri.stencilAttachment->image->getVkExtent().width;
				renderInfo.renderArea.extent.height = ri.stencilAttachment->image->getVkExtent().height;
			}	// otherwise let it be zero, as we dont render anything anyways

			renderInfo.layerCount = 1;	// Not sure what this does

			renderInfo.colorAttachmentCount = ri.colorAttachments.size();
			renderInfo.pColorAttachments = renderAttachmentBuffer.data();
			renderInfo.pDepthAttachment = depthAttachmentInfo.has_value() ? &depthAttachmentInfo.value() : nullptr;
			renderInfo.pStencilAttachment = stencilAttachmentInfo.has_value() ? &stencilAttachmentInfo.value() : nullptr;

			this->vkCmdBeginRenderingKHR(cmd, (VkRenderingInfoKHR*)&renderInfo);

			renderAttachmentBuffer.clear();

			setViewport(VkViewport{
				.x = 0,
				.y = 0,
				.width = (f32)renderInfo.renderArea.extent.width,
				.height = (f32)renderInfo.renderArea.extent.height,
				.minDepth = 0,
				.maxDepth = 1,
			});
		}
		void CommandList::endRendering() {
			DAXA_ASSERT_M(usesOnGPU == 0, "can not change command list, that is currently used on gpu");
			operationsInProgress -= 1;
			this->vkCmdEndRenderingKHR(cmd);
			boundPipeline = std::nullopt;
		}

		void CommandList::bindPipeline(GraphicsPipelineHandle& graphicsPipeline) {
			DAXA_ASSERT_M(usesOnGPU == 0, "can not change command list, that is currently used on gpu");
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline->getVkPipeline());
			usedGraphicsPipelines.push_back(graphicsPipeline);

			boundPipeline = BoundPipeline{
				.bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
				.layout = graphicsPipeline->getVkPipelineLayout(),
			};
		}

		void CommandList::reset() {
			DAXA_ASSERT_M(usesOnGPU == 0, "can not change command list, that is currently used on gpu");
			DAXA_ASSERT_M(operationsInProgress == 0, "can not reset command list with recordings in progress");
			empty = true;
			vkResetCommandPool(device, cmdPool, VkCommandPoolResetFlagBits::VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
			usedBuffers.clear();
			usedGraphicsPipelines.clear();
			usedImages.clear();
			usedSets.clear();
			boundPipeline.reset();
			usedStagingBuffers.clear();
		}

		void CommandList::setViewport(VkViewport const& viewport) {
			DAXA_ASSERT_M(usesOnGPU == 0, "can not change command list, that is currently used on gpu");
			vkCmdSetViewport(cmd, 0, 1, &viewport); 
			VkRect2D scissor{
				.offset = { 0, 0 },
				.extent = { (u32)viewport.width, (u32)viewport.height },
			};
			setScissor(scissor);
		}

		void CommandList::setScissor(VkRect2D const& scissor) {
			DAXA_ASSERT_M(usesOnGPU == 0, "can not change command list, that is currently used on gpu");
			vkCmdSetScissor(cmd, 0, 1, &scissor);
		}
		
		void CommandList::draw(u32 vertexCount, u32 instanceCount, u32 firstVertex, u32 firstInstance) {
			DAXA_ASSERT_M(usesOnGPU == 0, "can not change command list, that is currently used on gpu");
			vkCmdDraw(cmd, vertexCount, instanceCount, firstVertex, firstInstance);
		}

		void CommandList::bindSet(u32 setBinding, BindingSetHandle set) {
			DAXA_ASSERT_M(usesOnGPU == 0, "can not change command list, that is currently used on gpu");
			DAXA_ASSERT_M(boundPipeline.has_value(), "can not bind descriptor sets if there is no pipeline bound");
			vkCmdBindDescriptorSets(cmd, boundPipeline->bindPoint, boundPipeline->layout, setBinding, 1, &set->set, 0, nullptr);
			usedSets.push_back(set);
		}

		void CommandList::insertBarriers(std::span<MemoryBarrier> memBarriers, std::span<BufferBarrier> bufBarriers, std::span<ImageBarrier> imgBarriers) {
			DAXA_ASSERT_M(usesOnGPU == 0, "can not change command list, that is currently used on gpu");
			std::array<VkMemoryBarrier2KHR, 32> memBarrierBuffer;
			u32 memBarrierBufferSize = 0;
			std::array<VkBufferMemoryBarrier2KHR, 32> bufBarrierBuffer;
			u32 bufBarrierBufferSize = 0;
			std::array<VkImageMemoryBarrier2KHR, 32> imgBarrierBuffer;
			u32 imgBarrierBufferSize = 0;

			for (auto& barrier : memBarriers) {
				DAXA_ASSERT_M(memBarrierBufferSize < 32, "can only insert 32 barriers of one kind in a single insertBarriers call");
				memBarrierBuffer[memBarrierBufferSize++] = VkMemoryBarrier2KHR{
					.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR,
					.pNext = nullptr,
					.srcStageMask = barrier.awaitedStages,
					.srcAccessMask = barrier.awaitedAccess,
					.dstStageMask = barrier.waitingStages,
					.dstAccessMask = barrier.waitingAccess,
				};
			}

			for (auto& barrier : bufBarriers) {
				DAXA_ASSERT_M(bufBarrierBufferSize < 32, "can only insert 32 barriers of one kind in a single insertBarriers call");
				bufBarrierBuffer[bufBarrierBufferSize] = VkBufferMemoryBarrier2KHR{
					.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2_KHR,
					.pNext = nullptr,
					.srcStageMask = barrier.awaitedStages,
					.srcAccessMask = barrier.awaitedAccess,
					.dstStageMask = barrier.waitingStages,
					.dstAccessMask = barrier.waitingAccess,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.buffer = barrier.buffer->getVkBuffer(),
					.offset = barrier.offset,
					.size = barrier.buffer->getSize(),
				};

				usedBuffers.push_back(barrier.buffer);

				if (barrier.size.has_value()) {
					bufBarrierBuffer[bufBarrierBufferSize].size = *barrier.size;
				}

				bufBarrierBufferSize++;
			}

			for (auto& barrier : imgBarriers) {
				DAXA_ASSERT_M(bufBarrierBufferSize < 32, "can only insert 32 barriers of one kind in a single insertBarriers call");

				imgBarrierBuffer[imgBarrierBufferSize] = VkImageMemoryBarrier2KHR{
					.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR,
					.pNext = nullptr,
					.srcStageMask = barrier.awaitedStages,
					.srcAccessMask = barrier.awaitedAccess,
					.dstStageMask = barrier.waitingStages,
					.dstAccessMask = barrier.waitingAccess,
					.oldLayout = barrier.layoutBefore,
					.newLayout = barrier.layoutAfter,
					.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
					.image = barrier.image->getVkImage(),
					.subresourceRange = barrier.subRange.value_or(VkImageSubresourceRange{
						.aspectMask = barrier.image->getVkAspect(),
						.baseMipLevel = 0,
						.levelCount = barrier.image->getVkMipmapLevels(),
						.baseArrayLayer = 0,
						.layerCount = barrier.image->getVkArrayLayers(),
					})
				};

				usedImages.push_back(barrier.image);

				imgBarrierBufferSize++;
			}

			VkDependencyInfoKHR dependencyInfo{
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
				.pNext = nullptr,
				.memoryBarrierCount = memBarrierBufferSize,
				.pMemoryBarriers = memBarrierBuffer.data(),
				.bufferMemoryBarrierCount = bufBarrierBufferSize,
				.pBufferMemoryBarriers = bufBarrierBuffer.data(),
				.imageMemoryBarrierCount = imgBarrierBufferSize,
				.pImageMemoryBarriers = imgBarrierBuffer.data(),
			};

			this->vkCmdPipelineBarrier2KHR(cmd, &dependencyInfo);
		}

		CommandListHandle::~CommandListHandle() {
			if (list && list.use_count() == 1) {
				if (auto recyclingSharedData = list->recyclingData.lock()) {
					printf("recycle\n");
					list->reset();
					auto lock = std::unique_lock(recyclingSharedData->mut);
					recyclingSharedData->zombies.push_back(std::move(list));
				}
			}
		}
		
		CommandListHandle::CommandListHandle(std::shared_ptr<CommandList> list)
			: list{ list }
		{}
	}
}
