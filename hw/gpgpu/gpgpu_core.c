/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "gpgpu.h"
#include "gpgpu_core.h"

void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    memset(warp, 0, sizeof(*warp));
    warp->thread_id_base = thread_id_base;
    warp->warp_id = warp_id;
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];
    warp->active_mask = (num_threads >= 32) ? 0xFFFFFFFF : ((1u << num_threads) - 1);

    for (int i = 0; i < num_threads && i < 32; i++) {
        GPGPULane *lane = &warp->lanes[i];
        lane->pc = pc;
        lane->gpr[0] = 0;
        lane->mhartid = MHARTID_ENCODE(block_id_linear, warp_id,
                                        thread_id_base + i);
        lane->active = true;
    }
}

int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    for (int lane_id = 0; lane_id < 32; lane_id++) {
        if (!(warp->active_mask & (1u << lane_id))) continue;
        if (!warp->lanes[lane_id].active) continue;

        GPGPULane *lane = &warp->lanes[lane_id];
        uint32_t cycles = 0;

        while (cycles < max_cycles) {
            uint32_t inst;
            memcpy(&inst, s->vram_ptr + lane->pc, sizeof(inst));

            uint32_t opcode = inst & 0x7F;
            uint32_t rd     = (inst >> 7) & 0x1F;
            uint32_t funct3 = (inst >> 12) & 0x7;
            uint32_t rs1    = (inst >> 15) & 0x1F;
            uint32_t rs2    = (inst >> 20) & 0x1F;

            if (inst == 0x00100073) {
                lane->active = false;
                break;
            }

            switch (opcode) {
            case 0x37: {
                uint32_t imm = inst & 0xFFFFF000;
                lane->gpr[rd] = imm;
                lane->pc += 4;
                break;
            }
            case 0x13: {
                int32_t imm_i = (int32_t)(inst & 0xFFF00000) >> 20;
                uint32_t shamt = (inst >> 20) & 0x1F;
                uint32_t src = lane->gpr[rs1];
                switch (funct3) {
                case 0:
                    lane->gpr[rd] = src + imm_i;
                    break;
                case 1:
                    lane->gpr[rd] = src << shamt;
                    break;
                case 7:
                    lane->gpr[rd] = src & imm_i;
                    break;
                default:
                    return -1;
                }
                lane->pc += 4;
                break;
            }
            case 0x33: {
                if (funct3 == 0 && (inst >> 25) == 0x00) {
                    lane->gpr[rd] = lane->gpr[rs1] + lane->gpr[rs2];
                    lane->pc += 4;
                } else {
                    return -1;
                }
                break;
            }
            case 0x23: {
                if (funct3 == 2) {
                    int32_t imm_s = ((inst >> 25) << 5) | ((inst >> 7) & 0x1F);
                    if (imm_s & 0x800) imm_s |= 0xFFFFF000;
                    uint32_t addr = lane->gpr[rs1] + imm_s;
                    memcpy(s->vram_ptr + addr, &lane->gpr[rs2], 4);
                    lane->pc += 4;
                } else {
                    return -1;
                }
                break;
            }
            case 0x73: {
                if (funct3 == 1 || funct3 == 2) {
                    uint32_t csr = (inst >> 20) & 0xFFF;
                    if (csr == CSR_MHARTID) {
                        lane->gpr[rd] = lane->mhartid;
                        lane->pc += 4;
                    } else {
                        return -1;
                    }
                } else {
                    return -1;
                }
                break;
            }
            default:
                return -1;
            }

            cycles++;
            if (cycles >= max_cycles) break;
        }
    }
    return 0;
}

int gpgpu_core_exec_kernel(GPGPUState *s)
{
    uint64_t kernel_addr = s->kernel.kernel_addr;
    uint32_t gx = s->kernel.grid_dim[0];
    uint32_t gy = s->kernel.grid_dim[1];
    uint32_t gz = s->kernel.grid_dim[2];
    uint32_t bx = s->kernel.block_dim[0];
    uint32_t by = s->kernel.block_dim[1];
    uint32_t bz = s->kernel.block_dim[2];

    uint32_t threads_per_block = bx * by * bz;
    uint32_t warp_size = s->warp_size;
    uint32_t block_id[3];
    GPGPUWarp warp;

    if (threads_per_block == 0 || gx == 0) return -1;

    uint32_t linear_id = 0;
    for (uint32_t iz = 0; iz < gz; iz++) {
        block_id[2] = iz;
        for (uint32_t iy = 0; iy < gy; iy++) {
            block_id[1] = iy;
            for (uint32_t ix = 0; ix < gx; ix++) {
                block_id[0] = ix;
                uint32_t num_warps = (threads_per_block + warp_size - 1) / warp_size;
                for (uint32_t wid = 0; wid < num_warps; wid++) {
                    uint32_t base = wid * warp_size;
                    uint32_t n = (base + warp_size <= threads_per_block)
                                     ? warp_size
                                     : (threads_per_block - base);
                    gpgpu_core_init_warp(&warp, kernel_addr, base, block_id,
                                         n, wid, linear_id);
                    int ret = gpgpu_core_exec_warp(s, &warp, 100000);
                    if (ret < 0) return ret;
                }
                linear_id++;
            }
        }
    }
    return 0;
}
