/*
 * Auto-generated trampoline stubs.
 * x86-64 System V ABI.
 */

.macro CUDA_TRAMP name
    .text
    .globl \name
    .type \name, @function
\name:
    jmp *real_\name(%rip)
    .size \name, .-\name
.endm

#include "cuda_trampolines.inc"
