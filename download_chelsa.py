import requests
import time
import os

# env_list = ['clt', 'cmi', 'hurs', 'pet', 'pr', 'rsds', 'sfcWind', 'tas', 'tasmax', 'tasmin', 'vpd']
env_list = ['cmi']
months = range(1, 13)
years = range(2000, 2020)

combined_list = []
for item1 in env_list:
    for item2 in months:
        for item3 in years:
            combined_list.append([item1, item2, item3])
print(f'maximum index: {len(combined_list) - 1}')
start_idx = input('start index for downloading: ')

if start_idx == '':
    start_idx = 0

dir_base = '/work/klok0126/Chelsa'
link_pattern = dict(
    dir_base = 'https://os.zhdk.cloud.switch.ch/envicloud/chelsa/chelsa_V2/GLOBAL/monthly', 
    clt = 'clt/CHELSA_clt_[MONTH]_[YEAR]_V.2.1.tif', 
    cmi = 'cmi/CHELSA_cmi_[MONTH]_[YEAR]_V.2.1.tif', 
    hurs = 'hurs/CHELSA_hurs_[MONTH]_[YEAR]_V.2.1.tif', 
    pet = 'pet/CHELSA_pet_penman_[MONTH]_[YEAR]_V.2.1.tif', 
    pr = 'pr/CHELSA_pr_[MONTH]_[YEAR]_V.2.1.tif', 
    rsds = 'rsds/CHELSA_rsds_[YEAR]_[MONTH]_V.2.1.tif', 
    sfcWind = 'sfcWind/CHELSA_sfcWind_[MONTH]_[YEAR]_V.2.1.tif', 
    tas = 'tas/CHELSA_tas_[MONTH]_[YEAR]_V.2.1.tif', 
    tasmax = 'tasmax/CHELSA_tasmax_[MONTH]_[YEAR]_V.2.1.tif', 
    tasmin = 'tasmin/CHELSA_tasmin_[MONTH]_[YEAR]_V.2.1.tif', 
    vpd = 'vpd/CHELSA_vpd_[MONTH]_[YEAR]_V.2.1.tif'
)

for [env, month, year] in sorted(combined_list)[int(start_idx): ]:
    
    if not os.path.exists(f'{dir_base}/{env}'):
        os.makedirs(f'{dir_base}/{env}')
    
    link_itr = f'{link_pattern[env]}'.replace('[MONTH]', f'{month:02d}').replace('[YEAR]', str(year))
    link_full = f'{link_pattern["dir_base"]}/{link_itr}'
    if os.path.exists(f'{dir_base}/{link_itr}'):
        print(f"{link_itr} has already existed.")
        break
    print(f'Asseccing to file: {link_itr}')
    response = requests.get(link_full)
    if response.status_code == 200:
        with open(f'{dir_base}/{link_itr}', 'wb') as file:
            print(f"Downloaded.")
            file.write(response.content)
        time.sleep(3)
    else:
        print(f"Can not download links：{link_itr}.")
    response = None
print("All links are downloaded.")
