from __future__ import print_function
import numpy as np
import tensorflow as tf


"""
    
    2/20/2019
    Position - Simple Linear and RNN models to estimate the next position of a moving point in space
    
"""


train_mode='train'
test_mode='test'


class model:

  def __init__(self,mode=None,Linear=True,params=None):
    self.in_sz = params['points'] # size of points 
    self.true_sz = params['true_sz']
    stack = False
    self.rnns = {}
    self.params=params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    self.sess=tf.Session(config=config)
    self.mode = mode
    ## 1.a Construct graph
    # inputs and true value placeholders
    self.points = tf.placeholder(tf.float32, [None, self.in_sz], name='points') 
    self.coor_true = tf.placeholder(tf.float32, [None, self.true_sz], name='coor_true') 
        
    if Linear:
      ## 1.b Construct graph
      self.beta = tf.get_variable("beta", None, tf.float32,
                                  tf.random_normal([self.in_sz, self.true_sz],stddev=0.01))
      self.u = tf.get_variable('bias', None, tf.float32, 
                                tf.random_normal([self.true_sz], stddev=0.01))
      self.eps = tf.get_variable("eps", None, 
                                  tf.float32,tf.random_normal([self.true_sz], stddev=0.01)) 
      self.out = tf.add(tf.add(tf.matmul(self.points, self.beta), self.u), self.eps)
      # metrics
      self.residuals  = self.out - self.coor_true 
      self.error = (self.residuals/self.coor_true)
      self.fitted = self.out/tf.log(self.out) # normalized fitted due to scale
      self.cost = tf.square(self.residuals) # use square error for cost function
      self.sse = tf.cumsum(self.cost) # sum squared error metric
      self.loss = tf.reduce_sum(self.cost)
      if mode == 'train':
          ## 3. Loss + optimization
        g_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.02
        self.lr = tf.train.exponential_decay(starter_learning_rate, g_step, 300000, 0.9, staircase=True)
        opti = tf.train.AdamOptimizer(self.lr)
        # compute and apply gradients
        self.LMts_ = opti.minimize(self.loss, global_step=g_step)
      else:
        self.ts_=None

    else:
      self.seqlen = params['seqlen']
      # 1st layer
      self.w = tf.get_variable('w', None, tf.float32, 
                            tf.random_normal([self.seqlen, self.true_sz], stddev=0.01) ) 
      self.b = tf.get_variable('b', None, tf.float32, 
                            tf.random_normal([self.true_sz], stddev=0.01))
      ## 2. create the model
      basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.seqlen,
                                                activation=tf.nn.tanh)   
      self.outs, self.states = tf.nn.static_rnn(basic_cell,
                                            tf.split(X,2,axis=1),
                                            dtype=tf.float32)   
      self.z = tf.add(tf.matmul(self.states.c,self.w), self.b)
      self.residuals = self.z - self.coor_true
      self.error = (self.residuals/self.coor_true) # normalized due to scale points variance 
      self.fitted = self.z/tf.log(self.z) # normalized fitted due to scale
      self.cost = tf.square(self.residuals)
      self.sse = tf.cumsum(self.cost) # tf.cumsum depreciated use tf.math.cumsum instead
      self.loss = tf.reduce_sum(self.cost)   

    if mode == 'train':
        ## 3. Loss + optimization
      g_step = tf.Variable(0, trainable=False)
      starter_learning_rate = 0.02
      self.lr = tf.train.exponential_decay(starter_learning_rate, g_step, 300000, 0.9, staircase=True)
      opti = tf.train.AdamOptimizer(lr)
      if stack:
        s_ts_ = opti.minimize(s_loss, global_step=g_step)

      else:
        # compute and apply gradients
        self.ts_ = opti.minimize(loss, global_step=g_step)
      else:
        self.ts_ = None   
            
    vars_to_save={v.name: v for v in
                  tf.trainable_variables() + tf.get_collection("bn_pop_stats") + tf.get_collection("bn_counts")}
    #summary_writer = tf.train.SummaryWriter("/home/pogo/QS/SingleTrackRNN.logs", sess.graph)
    self.saver=tf.train.Saver(var_list=vars_to_save,max_to_keep=self.params['max_checkpoints'])
    self.sess.run(tf.global_variables_initializer())
    
  def create_inference_graph(batch_size=1, n_steps=16, n_features=26, width=64):
    """
    Returns a dictionary containing graphs
    """
        
    input_ph = tf.placeholder(dtype=tf.float32,
                              shape=[batch_size, n_steps, n_features],
                              name='input')
    sequence_lengths = tf.placeholder(dtype=tf.int32,
                                      shape=[batch_size],
                                      name='input_lengths')
    previous_state_c = tf.get_variable(dtype=tf.float32,
                                        shape=[batch_size, width],
                                        name='previous_state_c')
    previous_state_h = tf.get_variable(dtype=tf.float32,
                                        shape=[batch_size, width],
                                        name='previous_state_h')
    previous_state = tf.contrib.rnn.LSTMStateTuple(previous_state_c, previous_state_h)

    # Transpose from batch major to time major
    input_ = tf.transpose(input_ph, [1, 0, 2])

    # Flatten time and batch dimensions for feed forward layers
    input_ = tf.reshape(input_, [batch_size*n_steps, n_features])

    # Three ReLU hidden layers
    layer1 = tf.contrib.layers.fully_connected(input_, width)
    layer2 = tf.contrib.layers.fully_connected(layer1, width)
    layer3 = tf.contrib.layers.fully_connected(layer2, width)

    # Unidirectional LSTM
    rnn_cell = tf.contrib.rnn.LSTMBlockFusedCell(width)
    rnn, new_state = rnn_cell(layer3, initial_state=previous_state)
    new_state_c, new_state_h = new_state

    # Final hidden layer
    layer5 = tf.contrib.layers.fully_connected(rnn, width)

    # Output layer
    output = tf.contrib.layers.fully_connected(layer5, ALPHABET_SIZE+1, activation_fn=None)

    # Automatically update previous state with new state
    state_update_ops = [
        tf.assign(previous_state_c, new_state_c),
        tf.assign(previous_state_h, new_state_h)
    ]
    with tf.control_dependencies(state_update_ops):
        logits = tf.identity(logits, name='logits')

    # Create state initialization operations
    zero_state = tf.zeros([batch_size, n_cell_dim], tf.float32)
    initialize_c = tf.assign(previous_state_c, zero_state)
    initialize_h = tf.assign(previous_state_h, zero_state)
    initialize_state = tf.group(initialize_c, initialize_h, name='initialize_state')

    return {
        'inputs': {
            'input': input_ph,
            'input_lengths': sequence_lengths,
        },
        'outputs': {
            'output': logits,
            'initialize_state': initialize_state,
        }
    }
    
   
  def feed_dict(self,batch,nextp):
      fd={}
      fd[self.points] = batch
      fd[self.coor_true] = nextp
      return fd

  def train(self,points,nextp):
    #Inputs needed by model for a report
    fd = self.set_feed_dict(points, nextp)
    lr = 0.0
    if self.mode==train_mode:
      req_list=[self.LMts_, self.out, self.sse, self.error, self.residuals, self.fitted] 
      ts, y_est, sqerr, err, resi, fitted = self.sess.run(req_list,feed_dict=fd)
    else: # leave out train steps
      req_list=[self.out,self.sse, self.error, self.residuals, self.fittted]
      y_est, sqerr, err, resi, fitted = self.sess.run(req_list,feed_dict=fd)
      lr = 0.0
    return y_est, sqerr, err, resi, fitted
    
  def save(self, path,i):
    save_path = self.saver.save(self.sess, path+str(i)+'.ckpt',write_meta_graph=False)
    
  def restore(self, path):
    self.saver.restore(self.sess, path)
         

 
  def abline(m_rows=1, n_cols=49):

    """
    Line
    agrs
    ----
    m_rows, n_cols
    Creates a batch of vectors using a initial location within a 2600x2600 m^2 window
    Returns either a batch of the same vector (if m_rows>1) or just the vector  
    """

    # intial postion - s and y_init
    s  = np.random.randint(-1300, high=1300,size=None) + (np.pi/3)*np.random.rand() - np.random.rand()
    y_init = np.random.randint(-1300, high=1300,size=None) + (np.pi/3)*np.random.rand() - np.random.rand()
    ds = s+10 # change in direction is 10 meters
    slope = np.random.randint(-11, high=11,size=None) + (np.pi/3)*np.random.rand() - np.random.rand()
    x_ = np.linspace(s, ds, n_cols)
    y_ = slope * x_ +  y_init

    if m_rows > 1:
      b = np.linspace(s, ds, n_cols)
      for i in range(m_rows):
        x_ = np.vstack([x_, b])
      a = np.linspace(s, ds, n_cols)
      for i in range(m_rows):
        y_ = np.vstack([y_, b])

    return x_, y_    
    
    
  def window(loc, window=3):
    """
    Arguments
    ---------
    # loc - array of [x,y] pairs; 
    # window - number of pairs slide over = num_of_pairs*size_of_window
    Returns
    -------
    # a - Sliding window array
    """   
    a = loc.flatten()
    shift = window - 1
    indexer = np.arange(loc.shape[0])[None, :window] + np.arange(loc.shape[0] - shift)[:, None]
    return a[indexer]

  def pointpair(cols=16, w=5):
    x, y = abline(m_rows=1, n_cols=cols)
    x,y = window(x, window=w), window(y, window=w)
    x,y = x[:, :-1], y[:, :-1]
    corrs = np.array((x,y)).T
    corr, corr_true = corrs[0], corrs[1]
    corr, corr_true = np.concatenate(np.split(corr, 2)), np.concatenate(np.split(corr_true, 2), axis=1)
    return corr, corr_true   
        
  def rcircle(m_rows=1, r_radius=False, r_origin=False, pair=True, n_cols=360):
        
    """

    Circle 
    Returns either a vector or a batch of circular coordinates
    shifted from the unit circle. 

    """

    if r_radius and r_origin:
      mu = r.uniform(-100, 100) # radial eye in miles
      sigma = r.normalvariate(0,10)
      A = r.normalvariate(mu, sigma)
      f = r.normalvariate(5, 0.001) # scan  second 
      phi = r.uniform(-1776,1776) # window in miles
    else:
      if r_radius:
        mu = r.uniform(-100, 100) # radial eye in miles
        sigma = r.normalvariate(0,10)
        A = r.normalvariate(mu, sigma)
        phi = 0 
        f = 1
      else:
        A = 1
        f = r.normalvariate(5, 0.001) # scan  second 
        phi = r.uniform(-1776,1776) # window in miles

    # vector for x and y
    theta = 0*f
    dtheta = 2*np.pi*f 
    trainy = np.linspace(theta, dtheta, n_cols)
    trainy = A*np.sin(trainy) + phi
    trainx = np.linspace(theta, dtheta,n_cols)
    trainx = A*np.cos(trainx) + phi

    # batch
    if m_rows > 1:
      # build window - this one is currently not working
      a = np.linspace(theta,dtheta,n_cols)
      a = np.cos(a)
      for i in range(m_rows):
        trainx = np.vstack([trainx, a])
      # ditto
      b = np.linspace(theta,dtheta, n_cols)
      b = np.sin(b)
      for i in range(m_rows):
        trainy = np.vstack([trainy, b])  

    if pair:
      idxa = int(r.uniform(0,354))
      idxb = idxa + 3
      trainx = trainx[idxa:idxb]
      trainy = trainy[idxa:idxb]
      corrs = np.array((trainx,trainy)).T
      x,y = trainx[:-1], trainy[:-1]
      corr, corr_true = corrs[0:2].flatten(), corrs[2]
    return corr, corr_true

    if pair == False:    
      return trainx, trainy

  def sinewave(r_radius=True, r_origin=True, n_cols=360):    
    if r_radius and r_origin:
      mu = r.uniform(1, 10) # 
      sigma = r.normalvariate(0,0.1)
      A = r.normalvariate(mu, sigma)
      f = r.normalvariate(0.2, 10) # 
      phi = r.uniform(-1861,1865) # 
    else:
      if r_radius:
        mu = r.uniform(1, 10) # 
        sigma = r.normalvariate(0,0.1)
        A = r.normalvariate(mu, sigma)
        phi = 0 
        f = 1
      else:
        A = 1
        f = r.normalvariate(0.2, 10) # 
        phi = r.uniform(-1861,1865) # 
    # vector for x and y
    theta = 0*f 
    dtheta = 2*np.pi*f 
    swave = np.linspace(theta, dtheta, n_cols)
    swave = A*np.sin(swave) + phi
    return swave

        
  def residualplot(ax, err, fitted):
    x1, y1, x2, y2 = np.split(err, 4, axis=1)
    ex, ey = np.concatenate([x1,x2]), np.concatenate([y1,y2])
    fx1, fy1, fx2, fy2 = np.split(fitted, 4, axis=1)
    fx, fy = np.concatenate([fx1, fx2]), np.concatenate([fy1, fy2])
    ax[0].scatter(ex, fx)
    ax[0].set_title('Residual v. Fitted (X)')
    ax[0].set_xlabel('X Residuals')
    ax[0].set_ylabel("X Fitted")
    ax[1].scatter(ey, fy)
    ax[1].set_title('Residual v. Fitted (Y)')
    ax[1].set_xlabel('Y Residuals')
    ax[1].set_ylabel("Y Fitted")
    ax[0].hold(True)
    ax[1].hold(True)
    plt.draw()
    return ax

    
    
