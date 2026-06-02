from PIL import Image
import numpy as np

p = "./data/REPLACE_WITH_LOCAL_PATH"
u16 = np.array(Image.open(p), dtype=np.uint16)
print("u16 min/max:", u16.min(), u16.max())
uniq = np.unique(u16)
print("uniq head:", uniq[:20], "count:", uniq.size)
print("zero ratio:", (u16==0).mean())
