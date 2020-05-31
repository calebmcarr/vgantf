import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np

from generator import *
from discriminator import discr
#test casing

#test generator
fg_model = foreground()
bg_model = background()
mask = fg_mask(fg_model)
# keras automatically fills Batches but tf.random.normal does not
noise = tf.random.normal([1,100])
gen_fg_vid = fg_model(noise,training=False)
gen_mask = mask(noise,training=False)
gen_bg_vid = bg_model(noise,training=False)
vid = video(gen_mask,gen_fg_vid,gen_bg_vid)
print(vid)

#test Discriminator
disc = discr()
disc_res = disc(vid,training=False)
print(disc_res)
