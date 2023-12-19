use rand_distr::{Normal, Distribution};
use ndarray::{Array, Array2, arr2};
use ndarray_rand::RandomExt;

fn main() {
  let network = Network::new(vec![2, 3, 1]);
}

#[derive(Clone, Debug)]
struct Network {
  num_layers: usize,
  layer_sizes: Vec<u32>,
  biases: Vec<Array2<f32>>,
  weights: Vec<Array2<f32>>,
}

impl Network {
  /// The list `sizes` contains the number of neurons in the
  /// respective layers of the network. For example, if the list
  /// was [2, 3, 1] then it would be a three-layer network, with the
  /// first layer containing 2 neurons, the second layer 3 neurons,
  /// and the third layer 1 neuron.  The biases and weights for the
  /// network are initialized randomly, using a Gaussian
  /// distribution with mean 0, and variance 1.  Note that the first
  /// layer is assumed to be an input layer, and by convention we
  /// won't set any biases for those neurons, since biases are only
  /// ever used in computing the outputs from later layers.
  pub fn new(sizes: Vec<u32>) -> Self {
    let _layers = sizes.len();
    // Randomly initialize biases and weights with Gausian distribution.
    // This random initialization gives our stochastic gradient descent
    //  algorithm a place to start from.
    let _biases = (sizes[1..])
      .iter()
      .map(|x| Array2::random((*x as usize, 1), Normal::new(0., 1.).unwrap()))
      .collect();

    let _weights = sizes.windows(2)
      .into_iter()
      .map(|pair| Array2::random((pair[1] as usize, pair[0] as usize), Normal::new(0., 1.).unwrap()))
      .collect();

    return Network {
      num_layers: _layers,
      layer_sizes: sizes,
      biases: _biases,
      weights: _weights,
    }
  }

  pub fn feed_forward(self, array: &Array2<f32>) -> Array2<f32> {
    let mut _array = array.clone(); // Clone the input array
    
    for (b, w) in self.biases.iter().zip(self.weights.iter()) {
      let dot_product: Array2<f32> = w.dot(&_array) + b;
      _array = sigmoid(&dot_product);
    }
    
    _array
  }

  pub fn stochastic_gradient_descent<T>(
    self,
    training_data: &(f32, f32),
    epochs: &Vec<f32>,
    mini_batch_size: usize,
    eta: f32, // learning rate
    test_data: Option<T>
  ) {

  }
}

/// Note that when the input z is a vector or Numpy array, Array applies the
/// function sigmoid elementwise, that is, in vectorized form.
pub fn sigmoid(array: &Array2<f32>) -> Array2<f32> {
  let res: Vec<f32> = array
    .iter()
    .map(|x| 1.0 / (1.0 + (-x).exp()))
    .collect();

  Array2::from_shape_vec((1, res.len()), res).unwrap()
}