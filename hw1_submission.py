import numpy as np
import matplotlib.pyplot as plt
import math

def process_mnist(ds, target):
  xy_pairs = []
  image_scale = lambda x: 2 * (x / 255) - 1
  image_scale_np_func = np.vectorize(image_scale)
  for data in ds:
    image = np.array(data['image']).flatten()
    image = image_scale_np_func(image)

    label = 0
    if target == None:
      label = data['label']
    else:
      if target == data['label']:
        label = 1
      else:
        label = -1

    xy_pair = (image, label)
    xy_pairs.append(xy_pair)
  return xy_pairs

def lr_gradient(x, y, w):
  dimensions = len(w)
  gradient = np.zeros(dimensions)
  u = w[0] + np.dot(x, w[1:])
  u *= y
  deriv_part = -math.exp(-u) / (1 + math.exp(-u))
  gradient[0] = y * deriv_part
  gradient[1:] = x * y * deriv_part
  return np.asarray(gradient)
    
def lr_gradient_batch(xy_pairs, w):
  dimensions = len(xy_pairs[0][0]) + 1
  gradient = np.zeros(dimensions)
  x = np.array([pair[0] for pair in xy_pairs])
  y = np.array([pair[1] for pair in xy_pairs])
  for pair in xy_pairs:
    gradient += lr_gradient(pair[0], pair[1], w)
  
  gradient = gradient / len(xy_pairs)
  return gradient

# loss is -ln sigma(y(w * x + b))
def lr_loss(xy_pairs, w):
  loss_sum = 0
  for i in range(0, len(xy_pairs)):
    #loop calculating e^-y(w * x + b)
    #starts as -yb
    exponent = math.exp(-xy_pairs[i][1] * w[0])
    for j in range(0, len(w) - 1):
      exponent *= math.exp(-xy_pairs[i][1] * w[j + 1] * xy_pairs[i][0][j])
    cur_loss = -math.log(1 / (1 + exponent))
    loss_sum += cur_loss
  return loss_sum / len(xy_pairs)

#xy_pairs: list of (d-dim vector, binary label (int))
#w: d+1 dim vector (list?)
def lr_error(xy_pairs, w):
  error = 0
  for i in range(0, len(xy_pairs)):
    #if our calculated function comes out positive/negative (w * x + b)
    fc = w[0]
    for j in range(0, len(w) - 1):
      fc += xy_pairs[i][0][j] * w[j + 1]
    if (fc > 0 and xy_pairs[i][1] < 0) or (fc <= 0 and xy_pairs[i][1] > 0):
      error += 1

  return error / len(xy_pairs)

def full_error(xy_pairs, ws):
  error = 0
  for i in range(0, len(xy_pairs)):
    predicted_digit = 0
    highest_one_v_rest = -10000
    for j in range(0, len(ws)):
      fc = ws[j][0]
      for k in range(0, len(ws[j]) - 1):
        fc += xy_pairs[i][0][k] * ws[j][k + 1]
      if fc > highest_one_v_rest:
        highest_one_v_rest = fc
        predicted_digit = j
    if predicted_digit != xy_pairs[i][1]:
      error += 1
  return error / len(xy_pairs)
  