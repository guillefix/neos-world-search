import cv2
import requests
import json
import pandas as pd
import asyncio
import aiohttp
import aiofiles
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from projection.equirectangular import render_image_np, deg2rad

# import clip_embeddings
# import imp; imp.reload(clip_embeddings)
from clip_embeddings import embed_text, embed_image
#%%

def plot_image_grid(images, grid_shape):
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=grid_shape,  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

r = requests.get('https://api.neos.com/api/sessions')
data = json.loads(r.text)

d = pd.DataFrame(data)

# d[pd.isna(d["thumbnail"])] #hmm wat
d = d.set_index("sessionId")
d = d.dropna(subset=["thumbnail"])
d
# d.columns
# fields = ["name", "tags", "activeUsers", "sessionBeginTime"]

#%%

def get_thumbnail_url(uri):
    protocol = uri.split(":")[0]
    if protocol == "neosdb":
        temp = uri.split(":")[1][3:]
        temp = temp.split(".")
        neosdbSignature = temp[0]
        return "https://cloudxstorage.blob.core.windows.net/assets/"+neosdbSignature
    elif protocol == "https":
        return uri
    else:
        raise Exception("Protocol "+protocol+" not supported")


list(d["thumbnail"])

image_urls = list(map(lambda x: get_thumbnail_url(x), d["thumbnail"]))

image_urls

#%% DOWNLOAD

async def downloadImage(url, format="webp", name=None):
    async with aiohttp.ClientSession() as session:
        if name is None:
            name = url.split("/")[-1]
        filename = './thumbnails/'+name+"."+format
        async with session.get(url) as resp:
            if resp.status == 200:
                f = await aiofiles.open(filename, mode='wb')
                await f.write(await resp.read())
                await f.close()
                return filename
            else:
                return None

def get_snapshots(path):
    snapshots = []
    img = cv2.imread(path)
    for yaw in np.arange(0,360,45):
        face_size = 1000
        pitch = 0
        fov_h = 90
        fov_v = 90
        rimg = render_image_np(deg2rad(pitch), deg2rad(yaw), \
                          deg2rad(fov_v), deg2rad(fov_h), \
                          face_size, img)
        snapshots.append(rimg)
    return snapshots

#%%

coros = [downloadImage(url) for url in image_urls]

downloaded_images = await asyncio.gather(*coros)

#%%
queries = ["forest", "city", "furry", "sci-fi", "space station interior", "grass"]
query_embeddings = embed_text(queries)
normalized_query_embeddings =query_embeddings / np.linalg.norm(query_embeddings,axis=1, keepdims=True)

def get_clip_scores(imagepath, plot_image=False):
    temp = get_snapshots(imagepath)
    image_embeddings = embed_image(temp, fromarray=True)
    image_embeddings = image_embeddings.T
    if plot_image:
        plot_image_grid(temp,(3,3))

    dots = np.dot(normalized_query_embeddings,image_embeddings/np.linalg.norm(image_embeddings,axis=0))
    dots = np.sum(dots,axis=1)
    return list(dots), queries[np.argmax(dots)]

#%%

scores, label = get_clip_scores(downloaded_images[5])

label

from multiprocessing import Pool

pool = Pool()                         # Create a multiprocessing Pool
results = pool.map(get_clip_scores, downloaded_images[:6])  # process data_inputs iterable with pool
print(results)

#%%
