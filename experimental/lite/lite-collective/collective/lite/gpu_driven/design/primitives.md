# GPU-driven primitives

本文描述当前 GPU-driven 实现已有的基础操作及组合方式。公共架构见
[AGENTS.md](../AGENTS.md)，AllGather 的选路与算法见 [allgather.md](allgather.md)。
这里的 primitive 是实现中的基础操作，不表示已经存在一个统一的 primitive 类或独立编译模块。
修改这些操作时，需要同时检查设备调用方、host 初始化、service、错误传播和资源释放。

## Execution model and source map

GPU 用户 kernel 发起 collective，CPU 在初始化时准备资源并启动 service。
设备函数本身不会创建新的 CTA，也不会启动子 kernel。通常由一个 CTA 协作，
其中一个线程提交任务或更新控制字；其他线程按算法参与复制和同步。
HostCooperative 可以使用完整 cooperative grid，benchmark 仍只启动一个 block。
同一个 handle 不允许并发执行不同 collective。

| 文件 | 已实现的基础操作 |
|---|---|
| [gpu_collectives.cuh](../gpu_collectives.cuh) | 地址计算、CTA copy、归约运算、epoch/slot 管理、等待、任务提交和完成检查 |
| [task_fifo.hpp](../task_fifo.hpp) | `LiteTask`、8-slot FIFO、system-scope acquire/release |
| [allgather_host_cooperative.cuh](../allgather_host_cooperative.cuh) | 按 cooperative group 分摊 copy |
| [service.hpp](../service.hpp) | CPU 消费任务、提交 DMA、检查 CUDA events、发布完成 |
| [host_staging_buffer.hpp](../host_staging_buffer.hpp) | GPU-private host staging buffer、stream-ordered put/wait、资源所有权 |
| [allgather_network.cuh](../allgather_network.cuh) | 网络 AllGather 的设备提交、等待和完成封装 |
| [network_protocol.hpp](../network_protocol.hpp) | CPU schedule 与调用 CTA 的 SM 阶段握手 |
| [network_service.cu](../network_service.cu) | GPU-private 网络调度、RDMA 数据/控制操作、ACK 和 slot 复用 |
| [host_context.hpp](../host_context.hpp) | 初始化、映射、service 生命周期与清理 |
| [ipc_output_registration.hpp](../ipc_output_registration.hpp) | IPC 输出区注册、交换与映射 |

原生 CPU-driven collective 不是 GPU-driven primitive 的调用入口。GPU-only 修改不应改变
`../../allgather_intranode.cu`、`../../allgather_multinode.cu` 或
`../../cpu_staging_channel.hpp`；共享的 NodeExchangeBuffer/CpuSwitch 仍需跨路径检查。

## Common control primitives

### Publish and observe

`liteStoreRelease()` 使用 `st.release.sys.global.u64` 发布控制字，
`liteLoadAcquire()` 使用 `ld.acquire.sys.global.u64` 读取控制字。
设备目标要求 sm_70+；控制位置是自然对齐的 64-bit word，按协议安排写入者。
CPU 对 FIFO/control 使用 acquire/release 原子读写。

CTA 的 `__syncthreads()` 或 cooperative group 的 `group.sync()` 负责参与线程的同步，
不能代替跨 CPU/GPU 的发布协议。多个线程搬运数据后，调用方必须按实现中的
barrier/system fence 顺序完成可见性处理，再发布 ready。copy helper 本身不发布 ready。

### Wait, failure and retirement

* `liteCollectiveWait()` 默认等待 `observed >= target`，适用于单调 epoch/ticket。
* `liteCollectiveWait<true>()` 等待精确相等。OrderedSmall 的复用 payload-slot flag
  使用这一模式，避免尺寸切换后将旧 payload 字节误认为更新的 epoch。
* 等待同时检查 FIFO error、`~0ULL` 错误标记及 `timeoutCycles`。失败时
  `liteCollectivePoison()` 发布错误状态，返回失败；不能在已经通信后切换协议重试。
* `liteCollectiveWaitReusable()` 检查所需的网络/group reusable 条件以及各 rank 的 done。
  NIC 完成并不等于本地 GPU/DMA reader 已完成。
* 网络等待封装在失败时还设置 `LiteNetworkControl::abort`，让 CPU schedule 停止等待。
  失败 handle 应在调用 stream 结束后销毁，不能继续复用。

### Addressing, epochs and slots

`mscclppDeviceCollectiveSlab/Ready/Done()` 根据 backend 计算 payload/control 地址。
Host memory、IPC、host RDMA 的 stride 和索引不同；不能把它们统一理解为连续的同一布局。
专用 AllGather 执行器还可能使用当前消息尺寸的 packed rows 或网络 schedule 给出的 offset。

`liteCollectiveBeginBlock()` 分配 collective epoch、选择 whole-message slot 并等待旧使用者退出；
chunk readiness 与 whole-message slot 分开管理。当前通用协议使用两个 whole-message slots。
这不意味着每条优化路径都用同样的双缓冲循环：网络 AllGather 的 epoch/slot 由独立
schedule 管理，不调用这个 BeginBlock 来替代其协议。

FIFO ticket、collective epoch、网络 chunk-ready 值是不同的编号，不能相互替换。

## Intranode Data Path

### Data flow and responsibilities

单机数据移动由调用路径选择 SM 或 DMA，两者可组合，但并非每个阶段都经过 FIFO。

* **SM 路径**：调用 kernel 的线程将输入写入可访问的中间 payload，发布 ready；
  消费线程等待 ready 后读取数据，完成复制或归约，最后发布 done。
  HostMapped/HostCooperative AllGather 使用映射的 host slab；通用 reduction 可使用
  mapped host 或 IPC payload。线程间同步和 system-scope 发布由执行器组织。
* **Host DMA 路径**：GPU 提交任务，CPU service 在独立 stream 上安排
  `source GPU -> pinned host row -> destination GPU`，self row 可直接 D2D。
  D2H/H2D 依赖由 stream-ordered readiness、wait 和 CUDA events 表达。
* **IPC AllGather 路径**：GPU 提交 copy task，CPU service 通过 IPC 映射向 peer
  输出区发起 DMA。数据不经过 host payload slab，控制仍需要 GPU 与 CPU service 协作。

CPU service 消费 descriptor 并推进 DMA，不是通用 CPU SIMD 归约执行器。
SM 完成数据准备时按具体路径发布 ready；不能一概描述为“复制后再提交一个 FIFO task”。
实际 FIFO 使用递增 submitted/completed/retired，详见下文，不使用二选一的抽象 head/tail 或 flag 方案。

### SM copy and reduction

`liteCollectiveCopyBlock(dst, src, bytes, activeThreads)` 在当前 CTA 内按线程步长分摊复制：
两端都满足 8-byte 对齐时复制 64-bit words，再处理 byte tail；否则逐 byte 复制。
`activeThreads` 可以限制参与线程数量，不会增加 CTA，也不会调度更多 SM。

`liteAllGatherGroupCopy(group, ...)` 使用 `group.thread_rank()` 和 `group.size()` 分摊复制。
传入 thread block 时只在该 block 内工作；传入 cooperative grid 时跨参与 blocks 工作。
该 helper 的 64-bit 访问依赖调用路径已经满足对齐要求；同步由调用方执行。

`liteApplyReduction<T>()` 提供 sum/min/max 元素运算。当前 reduction 的 staging 和结果计算
由设备线程配合 mapped host/IPC payload 完成，不能描述成通用 CPU SIMD reduction service。
其 chunk 选择由 `liteCollectiveChunkBytes()` 管理，与 AllGather 专用 chunk 策略不同。

### GPU-to-CPU task FIFO

数据流：`GPU descriptor -> mapped host FIFO -> CPU service -> DMA -> completion word -> GPU`。

* `LiteTaskFifo` 有 8 个 descriptor slots，采用单 producer / 单 CPU consumer 协议。
  多个 GPU 线程不能无协调地同时调用 `litePostTask()`。
* `litePostTask()` 先按 `submitted` 与 `retired` 判断是否有空位，再写 descriptor，
  最后 release 发布 `submitted + 1`，返回这个值作为 ticket。没有空间时等待，失败返回 0。
* CPU acquire 读取 submitted，取出 descriptor，提交对应工作。
  对异步 DMA，通过 `cudaEventQuery()` 确认所需 events 全部完成后，才发布 slot.completed。
* `liteWaitTask()` 等待对应 slot 的 completed 达到 ticket。
  CPU 只将连续完成的 ticket 前缀推进到 retired，允许 descriptor slot 再利用。
* submitted 表示任务已发布；completed 表示该任务完成；retired 表示 descriptor 可复用。
  **这些值都不能单独替代 payload 的 ready/done、网络 ACK 或 slot-reuse 协议。**

当前 descriptor 类型如下。它们并非所有 AllGather 路径都会使用：

| Task kind | CPU service 的职责 |
|---|---|
| `Stage` | 提交到 staging buffer 的复制，完成后发布 staging readiness |
| `Gather` / `GatherPacked` | 从 staging 布局向输出提交收集复制 |
| `IpcCopySelf` | 本地输入到输出中 self row 的 D2D copy |
| `IpcPush` | 通过已注册输出的 offset，向 next rank 映射的输出区提交 D2D copy |
| `HostAllGather` | 执行一次完整 host-memory AllGather 的 DMA 提交与完成跟踪 |
| `NetworkAllGather` | 调用 GPU-private 网络 schedule 完成一次网络 AllGather |

### Host-memory DMA

数据流：`source GPU -> pinned host rank row -> peer output GPU`，self 可以直接 D2D。
`enqueueDeviceHostAllGather()` 使用 buffer.put 提交逐 chunk D2H，并按 stream 顺序发布 ready。
接收 streams 通过 buffer.wait 等待来源 rank/chunk 就绪，再提交 H2D。

左右两个 peer 范围分别组织批量复制。容量 stride 与当前输出 row stride 不相等时使用
`cudaMemcpy2DAsync()`，不能把 host slab 当作当前尺寸紧密排列的输出直接复制。
四个 service streams 分担 staging、peer receive 和 self-copy，完成检查覆盖全部四个 stream。
如果启用 self SM copy，由设备执行器负责那部分复制及其同步。

### CUDA IPC DMA

当前 IPC AllGather 不是“SM 先拷到 IPC staging buffer，再由 peer SM 读取”的实现。
输出区通过 `mscclppRegisterDeviceCollectiveIpcOutput()` 在调用前集体注册并映射。
`IpcPush` descriptor 中的 destination 是本地注册输出的地址；service 校验范围后，
将相对 offset 加到 next rank 的映射基址，提交 DMA。

GPU 发布 task，等待 CUDA event 对应的 task completion，之后才能发布 hop ready。
IPC ring 的 admission、hop readiness 和最终 done 由执行器维护。
输出注册不是每次设备 collective 都要做；替换、注销和释放前必须确保用户已完成。

## Internode Data Path

### Data flow and responsibilities

`source GPU -> local pinned send buffer -> local NIC -> remote NIC -> remote pinned receive buffer -> destination GPU`

GPU 发起调用并参与需要的 SM 阶段；CPU service/schedule 提交 IB 工作并推进传输。
所有跨节点 payload 经过 NIC 注册的 host memory，不使用 GDR。

1. **准备资源**：host 初始化发现节点/rank、GPU/NIC 分组，分配 staging/control，
   注册 NIC memory、交换远端地址和 key、建立连接，准备 FIFO、streams 和 events。
   控制区需满足设备访问要求；payload 是否映射决定相应 SM 阶段是否可用，不能假定所有路径都要求 mapped payload。
2. **准备发送数据**：按路径由 CTA 写入 mapped host payload，或由 CPU 提交 D2H。
   发布 readiness 前确保相应 SM/DMA 工作完成。网络 AllGather 提交的是整次调用，
   chunk/slot/offset 等工作由 schedule 组织，并非每个 chunk 都单独提交 FIFO descriptor。
3. **本地汇集并发送**：node/group leader 等待本地贡献，再发送对应 host 数据块。
   AllGather 汇集各 rank 数据，不执行归约；通用 reduction 的计算仍在 GPU，不能将
   CPU SIMD reduction 当作当前网络 primitive 的统一步骤。
4. **发布远端 readiness**：按路径的数据/控制传输顺序发布 epoch 或 chunk-ready。
   接收端观察控制值后才能消费 payload；多连接/多 rail 要覆盖所有相关传输。
5. **消费并退休**：CTA SM 或 CPU 提交的 H2D 将数据送入输出；实际消费完成后，
   再按路径推进 done、ACK 和 slot reuse。发送完成不意味着接收 GPU 已消费。

下面说明这些阶段使用的具体控制接口。

### Invocation and optional CTA phases

`liteNetworkAllGatherPost()` 由 CTA 的一个线程提交 whole-invocation `NetworkAllGather` task。
`networkPath` 指定执行器选定的路径，CPU 使用 GPU-private 网络 context 执行其 schedule。
这不是把每个网络 chunk 都转成通用 Stage/Gather task 的循环。

对于 OrderedSmall，CPU 通过 `LiteNetworkControl` 提供 slab/control 地址、offset、epoch
以及 stageWithSm/receiveWithSm 等字段，最后 release 发布 prepared=ticket。
调用 CTA 观察 prepared 后执行要求的 SM packing/receive 阶段，再发布 deviceDone=ticket。
CPU 等待这些阶段和 transport 输出完成后发布 task completion；
`liteNetworkAllGatherFinish()` 等待这个完成并将结果同步给 CTA。
prepared/deviceDone 使用 FIFO ticket，descriptor 内的 epoch 使用算法自己的编号。

### Staging, transfer and remote readiness

* D2H readiness 必须在 DMA 实际完成之后发布。发送方 NIC 和本地 receive consumers
  都必须观察各自所需的 readiness；远端 rdmaReady 不代表本节点其他 rank 的 D2H 已完成。
* CPU leader/group leader 发送已经准备好的 host 数据。单 slab、NUMA 分组及 pipeline
  使用各自 schedule 的 chunk、连接和控制布局，不能强行套用同一套 buffer-slot 协议。
* `signalRdmaReadyAtomic()` 是当前 generic/NUMA 调度的远端 ready 发布操作，
  使用 RDMA atomic 按上次已发布 epoch 的差值推进控制字；不是 CPU 线程之间抢占计数器。
  数据传输与控制发布的顺序、连接完成处理必须作为一个协议维护。
* Ordered-slot 和 one-rank pipeline 使用各自的数据/flag 写入协议，并非所有远端 ready
  都使用 atomic。Pipeline 为每个 chunk 保留稳定的 flag source，防止 NIC 异步读取时
  source 已被覆盖为后一个 chunk 的 ready 值。
* 多 rail 的 readiness 必须覆盖各 rail 的数据传输；只等待其中一条连接不能允许接收方复制。

### Receive, ACK and buffer reuse

接收方等待对应远端控制值，再使用 SM 或 H2D 消费 receive slab。
只有要求的 GPU/DMA consumers 完成，才能按路径协议发布 done/ACK 或释放 slot。
发送端完成写入、接收端观察 ready、接收端消费结束，是三个不同事件。

Generic、NUMA 和 pipeline schedules 分别管理 epoch、chunk readiness、ACK 及 slot reuse。
它们最终通过 FIFO completion 向用户 CTA 返回整次调用完成。
不能用 FIFO dequeue、单个 CUDA event 或一次 RDMA flush 代替整个调用完成条件。

### Progress and overlap

Pipeline 通过独立 buffers/slots 和明确的依赖重叠 D2H、RDMA、H2D。
仅增加 GPU chunk flags 不会自动产生网络重叠；CPU schedule 必须按路径推进 chunk，
并保留该路径的窗口大小、ACK 和 slot-reuse 条件。具体 chunk 参数见 AllGather 设计。

Service stream 的进展独立于调用 kernel，不能等待正在等待它的用户 kernel 结束。
网络调度必须兼顾发送、接收和 ACK，不能以单方向无限等待阻断所需的另一方向进展。
GPU-to-CPU 发布与 NIC-to-receiver readiness 是不同的边界，分别遵守对应协议；
CTA barrier 或 plain volatile flag 不能替代这些可见性条件。

## Resource lifecycle and implementation boundaries

初始化在 host 上完成 FIFO 映射、payload/control 资源准备、IPC/NIC 注册、连接交换、
非阻塞 service streams 和 events 创建。IPC-preferred AllGather 的 host fallback 也提前准备，
在发布通信前选路。容量、拓扑或运行时错误不允许各 rank 独立中途换协议。

CPU service 不等待发起 collective 的用户 kernel 完成，也不为设备调用启动 child kernel。
释放资源前先通知停止/abort、join workers、drain 已提交复制，再释放 events、注册和 buffers。
详细 ownership 以 `DeviceCollectiveContext::releaseResources()` 为准。

本文是源码行为说明，不是硬件正确性或性能验证结论。修改 primitive 后需验证映射开关、
对齐/tail、in-place、FIFO wrap、混合消息尺寸、epoch/slot reuse、各参与者错误传播，
并在对应单机/多机 GPU 环境运行 collective correctness preflight。
