#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

#include "opencl_cl_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "opencl_runtime.h"
#include "opencl_kernels.h"

typedef struct {
    int sx;
    int sy;
    int sz;
    int bord;
    float dx;
    float dy;
    float dz;
    float dt;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel_propagate;
    cl_kernel kernel_insert;
    cl_device_id device;
    cl_mem ch1dxx;
    cl_mem ch1dyy;
    cl_mem ch1dzz;
    cl_mem ch1dxy;
    cl_mem ch1dyz;
    cl_mem ch1dxz;
    cl_mem v2px;
    cl_mem v2pz;
    cl_mem v2sz;
    cl_mem v2pn;
    cl_mem pp;
    cl_mem pc;
    cl_mem qp;
    cl_mem qc;
    size_t global[2];
    size_t local[2];
    int use_local_size;
} OpenCLState;

static OpenCLState state = {0};

#define OPENCL_CHECK(err, msg)                                                     \
    do {                                                                           \
        if ((err) != CL_SUCCESS) {                                                 \
            fprintf(stderr, "OpenCL error %s (code %d)\n", (msg), (int)(err));    \
            exit(EXIT_FAILURE);                                                    \
        }                                                                          \
    } while (0)

static void release_mem(cl_mem *buffer) {
    if (*buffer) {
        clReleaseMemObject(*buffer);
        *buffer = NULL;
    }
}

static void select_device(cl_device_id *device) {
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    OPENCL_CHECK(err, "clGetPlatformIDs count");
    if (num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found.\n");
        exit(EXIT_FAILURE);
    }

    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    if (!platforms) {
        fprintf(stderr, "Failed to allocate memory for platform list.\n");
        exit(EXIT_FAILURE);
    }

    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    OPENCL_CHECK(err, "clGetPlatformIDs list");

    cl_device_id chosen = NULL;
    for (cl_uint i = 0; i < num_platforms && chosen == NULL; ++i) {
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &chosen, NULL);
        if (err == CL_SUCCESS) {
            break;
        }
    }

    if (chosen == NULL) {
        for (cl_uint i = 0; i < num_platforms && chosen == NULL; ++i) {
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &chosen, NULL);
            if (err == CL_SUCCESS) {
                break;
            }
        }
    }

    free(platforms);

    if (chosen == NULL) {
        fprintf(stderr, "No OpenCL device found.\n");
        exit(EXIT_FAILURE);
    }

    *device = chosen;
}

static void build_program(OpenCLState *s) {
    cl_int err = 0;
    const char *source = opencl_kernel_source;
    size_t length = strlen(opencl_kernel_source);

    s->program = clCreateProgramWithSource(s->context, 1, &source, &length, &err);
    OPENCL_CHECK(err, "clCreateProgramWithSource");

    err = clBuildProgram(s->program, 1, &s->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(s->program, s->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size + 1);
        if (log) {
            clGetProgramBuildInfo(s->program, s->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            log[log_size] = '\0';
            fprintf(stderr, "OpenCL build log:\n%s\n", log);
            free(log);
        }
        OPENCL_CHECK(err, "clBuildProgram");
    }

    s->kernel_propagate = clCreateKernel(s->program, "propagate", &err);
    OPENCL_CHECK(err, "clCreateKernel propagate");

    s->kernel_insert = clCreateKernel(s->program, "insert_source", &err);
    OPENCL_CHECK(err, "clCreateKernel insert_source");
}

static void create_and_write_buffer(cl_mem *buffer, cl_context context, cl_command_queue queue,
                                    cl_mem_flags flags, size_t bytes, const void *host_ptr) {
    cl_int err = 0;
    *buffer = clCreateBuffer(context, flags, bytes, NULL, &err);
    OPENCL_CHECK(err, "clCreateBuffer");
    if (host_ptr != NULL && bytes > 0) {
        err = clEnqueueWriteBuffer(queue, *buffer, CL_TRUE, 0, bytes, host_ptr, 0, NULL, NULL);
        OPENCL_CHECK(err, "clEnqueueWriteBuffer");
    }
}

void OPENCL_Initialize(const int sx, const int sy, const int sz, const int bord,
                       float dx, float dy, float dz, float dt,
                       float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz,
                       float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz,
                       float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
                       float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
                       float * restrict phi, float * restrict theta,
                       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{
    (void)vpz;
    (void)vsv;
    (void)epsilon;
    (void)delta;
    (void)phi;
    (void)theta;

    state.sx = sx;
    state.sy = sy;
    state.sz = sz;
    state.bord = bord;
    state.dx = dx;
    state.dy = dy;
    state.dz = dz;
    state.dt = dt;
    state.global[0] = (size_t)sx;
    state.global[1] = (size_t)sy;
    state.local[0] = 0;
    state.local[1] = 0;
    state.use_local_size = 0;

    select_device(&state.device);

    cl_int err = 0;
    state.context = clCreateContext(NULL, 1, &state.device, NULL, NULL, &err);
    OPENCL_CHECK(err, "clCreateContext");

    state.queue = clCreateCommandQueue(state.context, state.device, 0, &err);
    OPENCL_CHECK(err, "clCreateCommandQueue");

    build_program(&state);

    const size_t volume = (size_t)sx * (size_t)sy * (size_t)sz;
    const size_t bytes = volume * sizeof(float);

    create_and_write_buffer(&state.ch1dxx, state.context, state.queue, CL_MEM_READ_ONLY, bytes, ch1dxx);
    create_and_write_buffer(&state.ch1dyy, state.context, state.queue, CL_MEM_READ_ONLY, bytes, ch1dyy);
    create_and_write_buffer(&state.ch1dzz, state.context, state.queue, CL_MEM_READ_ONLY, bytes, ch1dzz);
    create_and_write_buffer(&state.ch1dxy, state.context, state.queue, CL_MEM_READ_ONLY, bytes, ch1dxy);
    create_and_write_buffer(&state.ch1dyz, state.context, state.queue, CL_MEM_READ_ONLY, bytes, ch1dyz);
    create_and_write_buffer(&state.ch1dxz, state.context, state.queue, CL_MEM_READ_ONLY, bytes, ch1dxz);
    create_and_write_buffer(&state.v2px, state.context, state.queue, CL_MEM_READ_ONLY, bytes, v2px);
    create_and_write_buffer(&state.v2pz, state.context, state.queue, CL_MEM_READ_ONLY, bytes, v2pz);
    create_and_write_buffer(&state.v2sz, state.context, state.queue, CL_MEM_READ_ONLY, bytes, v2sz);
    create_and_write_buffer(&state.v2pn, state.context, state.queue, CL_MEM_READ_ONLY, bytes, v2pn);
    create_and_write_buffer(&state.pp, state.context, state.queue, CL_MEM_READ_WRITE, bytes, pp);
    create_and_write_buffer(&state.pc, state.context, state.queue, CL_MEM_READ_WRITE, bytes, pc);
    create_and_write_buffer(&state.qp, state.context, state.queue, CL_MEM_READ_WRITE, bytes, qp);
    create_and_write_buffer(&state.qc, state.context, state.queue, CL_MEM_READ_WRITE, bytes, qc);

    clFinish(state.queue);

    char device_name[256];
    err = clGetDeviceInfo(state.device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    OPENCL_CHECK(err, "clGetDeviceInfo name");
    printf("OpenCL source using device %s.\n", device_name);
}

void OPENCL_Finalize()
{
    release_mem(&state.ch1dxx);
    release_mem(&state.ch1dyy);
    release_mem(&state.ch1dzz);
    release_mem(&state.ch1dxy);
    release_mem(&state.ch1dyz);
    release_mem(&state.ch1dxz);
    release_mem(&state.v2px);
    release_mem(&state.v2pz);
    release_mem(&state.v2sz);
    release_mem(&state.v2pn);
    release_mem(&state.pp);
    release_mem(&state.pc);
    release_mem(&state.qp);
    release_mem(&state.qc);

    if (state.kernel_propagate) {
        clReleaseKernel(state.kernel_propagate);
        state.kernel_propagate = NULL;
    }
    if (state.kernel_insert) {
        clReleaseKernel(state.kernel_insert);
        state.kernel_insert = NULL;
    }
    if (state.program) {
        clReleaseProgram(state.program);
        state.program = NULL;
    }
    if (state.queue) {
        clReleaseCommandQueue(state.queue);
        state.queue = NULL;
    }
    if (state.context) {
        clReleaseContext(state.context);
        state.context = NULL;
    }
    state.device = NULL;
}

void OPENCL_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
    if (!pc) {
        return;
    }
    const size_t volume = (size_t)sx * (size_t)sy * (size_t)sz;
    const size_t bytes = volume * sizeof(float);
    cl_int err = clEnqueueReadBuffer(state.queue, state.pc, CL_TRUE, 0, bytes, pc, 0, NULL, NULL);
    OPENCL_CHECK(err, "clEnqueueReadBuffer pc");
}

static void swap_buffers(cl_mem *a, cl_mem *b) {
    cl_mem tmp = *a;
    *a = *b;
    *b = tmp;
}

void OPENCL_Propagate(const int sx, const int sy, const int sz, const int bord,
                      const float dx, const float dy, const float dz, const float dt, const int it,
                      float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{
    (void)pp;
    (void)pc;
    (void)qp;
    (void)qc;
    cl_int err = 0;

    err  = clSetKernelArg(state.kernel_propagate, 0, sizeof(int), &sx);
    err |= clSetKernelArg(state.kernel_propagate, 1, sizeof(int), &sy);
    err |= clSetKernelArg(state.kernel_propagate, 2, sizeof(int), &sz);
    err |= clSetKernelArg(state.kernel_propagate, 3, sizeof(int), &bord);
    err |= clSetKernelArg(state.kernel_propagate, 4, sizeof(float), &dx);
    err |= clSetKernelArg(state.kernel_propagate, 5, sizeof(float), &dy);
    err |= clSetKernelArg(state.kernel_propagate, 6, sizeof(float), &dz);
    err |= clSetKernelArg(state.kernel_propagate, 7, sizeof(float), &dt);
    err |= clSetKernelArg(state.kernel_propagate, 8, sizeof(int), &it);
    err |= clSetKernelArg(state.kernel_propagate, 9, sizeof(cl_mem), &state.ch1dxx);
    err |= clSetKernelArg(state.kernel_propagate, 10, sizeof(cl_mem), &state.ch1dyy);
    err |= clSetKernelArg(state.kernel_propagate, 11, sizeof(cl_mem), &state.ch1dzz);
    err |= clSetKernelArg(state.kernel_propagate, 12, sizeof(cl_mem), &state.ch1dxy);
    err |= clSetKernelArg(state.kernel_propagate, 13, sizeof(cl_mem), &state.ch1dyz);
    err |= clSetKernelArg(state.kernel_propagate, 14, sizeof(cl_mem), &state.ch1dxz);
    err |= clSetKernelArg(state.kernel_propagate, 15, sizeof(cl_mem), &state.v2px);
    err |= clSetKernelArg(state.kernel_propagate, 16, sizeof(cl_mem), &state.v2pz);
    err |= clSetKernelArg(state.kernel_propagate, 17, sizeof(cl_mem), &state.v2sz);
    err |= clSetKernelArg(state.kernel_propagate, 18, sizeof(cl_mem), &state.v2pn);
    err |= clSetKernelArg(state.kernel_propagate, 19, sizeof(cl_mem), &state.pp);
    err |= clSetKernelArg(state.kernel_propagate, 20, sizeof(cl_mem), &state.pc);
    err |= clSetKernelArg(state.kernel_propagate, 21, sizeof(cl_mem), &state.qp);
    err |= clSetKernelArg(state.kernel_propagate, 22, sizeof(cl_mem), &state.qc);
    OPENCL_CHECK(err, "clSetKernelArg propagate");

    const size_t *local = state.use_local_size ? state.local : NULL;
    err = clEnqueueNDRangeKernel(state.queue, state.kernel_propagate, 2, NULL, state.global, local, 0, NULL, NULL);
    OPENCL_CHECK(err, "clEnqueueNDRangeKernel propagate");

    clFinish(state.queue);

    swap_buffers(&state.pp, &state.pc);
    swap_buffers(&state.qp, &state.qc);
}

void OPENCL_InsertSource(const float val, const int iSource, float *p, float *q)
{
    (void)p;
    (void)q;
    cl_int err = 0;
    const size_t global = 1;
    err  = clSetKernelArg(state.kernel_insert, 0, sizeof(float), &val);
    err |= clSetKernelArg(state.kernel_insert, 1, sizeof(int), &iSource);
    err |= clSetKernelArg(state.kernel_insert, 2, sizeof(cl_mem), &state.pc);
    err |= clSetKernelArg(state.kernel_insert, 3, sizeof(cl_mem), &state.qc);
    OPENCL_CHECK(err, "clSetKernelArg insert_source");

    err = clEnqueueNDRangeKernel(state.queue, state.kernel_insert, 1, NULL, &global, NULL, 0, NULL, NULL);
    OPENCL_CHECK(err, "clEnqueueNDRangeKernel insert_source");
    clFinish(state.queue);
}
