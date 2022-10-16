target='car'
other='aeroplane'
from duckduckgo_search import ddg_images
ddg_images(other,download=True,max_results=850)
ddg_images(target,download=True,max_results=850)
from PIL import Image
import os
folder_paths_target=os.listdir('./ddg_images_'+target+'_20221008_154406/')
for i in range (len(folder_paths_target)):
    try:
        image=Image.open('./ddg_images_car_20221008_154406/'+folder_paths_target[i])
        new_image=image.resize((500,500))
        new_image=new_image.convert('L')
        new_image.save('./dataset1/'+target+'_greyscale_'+str(i)+'.jpg')
    except:
        print(i)
    
    
    
folder_paths_target=os.listdir('./ddg_images_'+other+'_20221008_154605/')

for i in range (len(folder_paths_target)):
    try:
        image=Image.open('./ddg_images_aeroplane_20221008_154605/'+folder_paths_target[i])
        new_image=image.resize((500,500))
        new_image=new_image.convert('L')
        new_image.save('./dataset1/'+other+'_greyscale_'+str(i)+'.jpg')
    except:
        print(i)
     
    
    