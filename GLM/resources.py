import psutil
import subprocess as sp
import math
def get_resource_info():
    try:
        command= "nvidia-smi --query-gpu=memory.free,memory.total,pstate,compute_cap,utilization.gpu,count,gpu_name,driver_version --format=csv"
        smi_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        p_gpu_vals = smi_info[0].split(", ")
        tot_free_vram, tot_vram, perfomance_state, cuda_comp_ability, gpu_utilization, gpu_count, gpu_name, driver_version  = smi_info[0].split(", ")
        perfomance_state = round((12-int(perfomance_state[1:]))/12*100)
        tot_free_vram = int(tot_free_vram[:-4])
        tot_vram = int(tot_vram[:-4])
    except:
        tot_free_vram, tot_vram, perfomance_state, cuda_comp_ability, gpu_utilization, gpu_count, gpu_name, driver_version = [-1]*8
    
    vm = psutil.virtual_memory()
    total_cpu_mem = math.floor(vm.total / (1024 ** 2))
    used_cpu_mem = math.floor(vm.used / (1024 ** 2))
    available_cpu_mem = total_cpu_mem-used_cpu_mem
    cpu_load_avg_1m, cpu_load_avg_5m, cpu_load_avg_15m = [round(i,2) for i in psutil.getloadavg()]
    
    cur_ps = psutil.Process()
    with cur_ps.oneshot():
        proc_cpu_time_total = round(sum(cur_ps.cpu_times()),3)
        proc_ram_usage = round(cur_ps.memory_info().rss/1024**2)
    perfomance_dic = {}
    vars = "tot_free_vram, tot_vram, perfomance_state, cuda_comp_ability, gpu_utilization, gpu_count, gpu_name, driver_version, total_cpu_mem, used_cpu_mem, "\
    "available_cpu_mem, cpu_load_avg_1m, cpu_load_avg_5m, cpu_load_avg_15m, proc_cpu_time_total, proc_ram_usage"
    
    for i in vars.split(", "):
        perfomance_dic[i] = locals()[i]
    return perfomance_dic


def get_resource_diff(res_a, res_b):
    vram_diff = res_a["tot_free_vram"]- res_b["tot_free_vram"]
    ram_diff = res_a["available_cpu_mem"] - res_b["available_cpu_mem"]
    cpu_time_diff = res_b["proc_cpu_time_total"]-res_a["proc_cpu_time_total"]
    load_avg_1m_diff = res_a["cpu_load_avg_1m"]- res_b["cpu_load_avg_1m"]
    diff_dic = {"vram_diff": vram_diff, "ram_diff":ram_diff, "cpu_time_diff": cpu_time_diff, "load_avg_1m_diff": load_avg_1m_diff}
    return diff_dic