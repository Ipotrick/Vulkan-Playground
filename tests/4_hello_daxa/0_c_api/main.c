#include <daxa/c/daxa.h>
#include <stdio.h>
#include <stdlib.h>

void handle_daxa_result(daxa_Result res)
{
    if (res != DAXA_RESULT_SUCCESS)
    {
        printf("DAXA ERROR: %d\n", (int)res);
        exit(-1);
    }
}

int main(void)
{
    daxa_Instance instance = {0};
    daxa_InstanceInfo instance_info = DAXA_DEFAULT_INSTANCE_INFO;
    handle_daxa_result(daxa_create_instance(&instance_info, &instance));

    daxa_Device device = {0};
    daxa_DeviceInfo2 device_info = DAXA_DEFAULT_DEVICE_INFO_2;

    // Manual Selection:
    // daxa_DeviceProperties const* devices = {0};
    // daxa_u32 device_count = {0};
    // daxa_instance_list_devices_properties(instance, &devices, &device_count);

    // daxa_u32 choosen_device_index = ~0u;
    // for (daxa_u32 i = 0; i < device_count; ++i)
    // {
    //     daxa_DeviceProperties const * device_properties = devices + i;
    //     daxa_Bool8 const suitable = 
    //         device_properties->problems == DAXA_MISSING_REQUIRED_VK_FEATURE_NONE &&
    //         device_properties->mesh_shader_properties.has_value;
    //     if (suitable)
    //     {`
    //         choosen_device_index = i;
    //         break;
    //     }
    // }
    // if (choosen_device_index == ~0u)
    // {
    //     printf("found no supported device\n");
    //     exit(-1);
    // }
    // else
    // {
    //     printf("choose device with name \"%s\".\n", devices[choosen_device_index].device_name);
    // }
    // device_info.physical_device_index = choosen_device_index;

    // Automatic Selection:
    handle_daxa_result(daxa_instance_choose_device(instance, DAXA_IMPLICIT_FEATURE_FLAG_NONE, &device_info));

    handle_daxa_result(daxa_instance_create_device_2(instance, &device_info, &device));

    daxa_dvc_dec_refcnt(device);
    daxa_instance_dec_refcnt(instance);
    return 0;
}
