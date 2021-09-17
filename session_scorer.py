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
# from multiprocessing import Pool
from aiomultiprocess import Pool

from projection.equirectangular import render_image_np, deg2rad

# import clip_embeddings
# import imp; imp.reload(clip_embeddings)
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

async def get_snapshots(path):
    if path is not None:
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
    else:
        return None

#%%



async def main():

    print("downloading")
    coros = [downloadImage(url) for url in image_urls]

    downloaded_images = await asyncio.gather(*coros)
    print("downloaded")

    await asyncio.sleep(1)
    #%%
    print("starting loop thing")
    async with Pool() as pool:
        async for snapshots in pool.map(get_snapshots, downloaded_images):  # process data_inputs iterable with pool
            print("hi")
            if snapshots is not None:
                scores, label = get_clip_scores(snapshots)
                print(label)
            else:
                print("None")

    #%%

if __name__ == '__main__':
    from clip_embeddings import embed_text, embed_image
    #%%
    queries = ["forest", "city", "furry", "sci-fi", "space station interior", "grass"]
    query_embeddings = embed_text(queries)
    normalized_query_embeddings =query_embeddings / np.linalg.norm(query_embeddings,axis=1, keepdims=True)

    # temp = get_snapshots(imagepath)
    def get_clip_scores(snapshots, plot_image=False):
        image_embeddings = embed_image(snapshots, fromarray=True)
        image_embeddings = image_embeddings.T
        if plot_image:
            plot_image_grid(temp,(3,3))

        dots = np.dot(normalized_query_embeddings,image_embeddings/np.linalg.norm(image_embeddings,axis=0))
        dots = np.sum(dots,axis=1)
        return list(dots), queries[np.argmax(dots)]

    # loop = asyncio.get_event_loop()
    # loop = asyncio.new_event_loop()
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
    # loop.run_until_complete(main())

    # loop.close()
