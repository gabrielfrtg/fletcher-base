#ifndef OPENCL_STUBS_H
#define OPENCL_STUBS_H

#include <stddef.h>
#include <stdint.h>

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_bitfield cl_device_info;
typedef cl_bitfield cl_program_build_info;
typedef cl_bitfield cl_mem_flags;
typedef uintptr_t cl_bool;
typedef intptr_t cl_context_properties;

typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_event *cl_event;
typedef struct _cl_sampler *cl_sampler;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_DEFAULT (1ULL << 0)
#define CL_DEVICE_TYPE_CPU (1ULL << 1)
#define CL_DEVICE_TYPE_GPU (1ULL << 2)

#define CL_DEVICE_NAME 0x102B

#define CL_MEM_READ_WRITE (1ULL << 0)
#define CL_MEM_WRITE_ONLY (1ULL << 1)
#define CL_MEM_READ_ONLY (1ULL << 2)

#define CL_FALSE 0
#define CL_TRUE 1

#define CL_PROGRAM_BUILD_LOG 0x1183

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms);
cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries,
                      cl_device_id *devices, cl_uint *num_devices);
cl_context clCreateContext(const cl_context_properties *properties, cl_uint num_devices,
                           const cl_device_id *devices, void (*pfn_notify)(const char *, const void *, size_t, void *),
                           void *user_data, cl_int *errcode_ret);
cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device,
                                      cl_command_queue_properties properties, cl_int *errcode_ret);
cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings,
                                     const size_t *lengths, cl_int *errcode_ret);
cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list,
                      const char *options, void (*pfn_notify)(cl_program, void *), void *user_data);
cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name,
                             size_t param_value_size, void *param_value, size_t *param_value_size_ret);
cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret);
cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret);
cl_int clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write,
                            size_t offset, size_t size, const void *ptr, cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list, cl_event *event);
cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
                           size_t offset, size_t size, void *ptr, cl_uint num_events_in_wait_list,
                           const cl_event *event_wait_list, cl_event *event);
cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);
cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
                              const size_t *global_work_offset, const size_t *global_work_size,
                              const size_t *local_work_size, cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list, cl_event *event);
cl_int clFinish(cl_command_queue command_queue);
cl_int clReleaseMemObject(cl_mem memobj);
cl_int clReleaseKernel(cl_kernel kernel);
cl_int clReleaseProgram(cl_program program);
cl_int clReleaseCommandQueue(cl_command_queue command_queue);
cl_int clReleaseContext(cl_context context);
cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size,
                       void *param_value, size_t *param_value_size_ret);

#endif
