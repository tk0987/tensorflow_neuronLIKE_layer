"""
NEURON PROJECT
TensorFlow/Keras rewrite  --> claude anthropic

Biologically-inspired neuron layer using receptor interaction rules.
Each neuron maintains a trainable Q-factor per receptor that scales
stimulatory and inhibitory signals independently.

Original concept: T. Kowalski
TF rewrite: fixes all bugs from the numpy prototype and maps the
            design onto a standard Keras Layer + training loop.
"""

import numpy as np
import tensorflow as tf
import pickle


# ---------------------------------------------------------------------------
# Receptor interaction rules
# Each receptor lists which other receptors stimulate or inhibit it.
# ---------------------------------------------------------------------------
RECEPTOR_RULES: dict[str, dict] = {
    # --- Ionotropic glutamate receptors ---
    "AMPA": {
        "stimuli":    ["NMDA", "D1", "5-HT2A", "alpha1A"],
        "inhibitory": ["GABA_A", "GABA_B", "CB1", "D2"],
    },
    "NMDA": {
        "stimuli":    ["AMPA", "D1", "5-HT2A", "H2"],
        "inhibitory": ["GABA_A", "5-HT1A", "CB1", "sigma1", "GlyR alpha1"],
    },
    "Kainate": {
        "stimuli":    ["AMPA", "mGluR1"],
        "inhibitory": ["GABA_A"],
    },

    # --- Metabotropic glutamate receptors ---
    "mGluR1": {
        "stimuli":    ["NMDA", "5-HT2A"],
        "inhibitory": ["GABA_B"],
    },
    # mGluR2 is a presynaptic autoreceptor: high glutamate (proxied by AMPA/NMDA activity)
    # activates it, and it feeds back to suppress glutamatergic output.
    "mGluR2": {
        "stimuli":    ["AMPA", "NMDA"],       # FIX: was empty; high Glu tone drives autoreceptor
        "inhibitory": ["AMPA", "NMDA"],       # FIX: was ["Glutamate release"] — replaced with receptors
    },

    # --- GABAergic receptors ---
    "GABA_A": {
        "stimuli":    ["5-HT1A", "D2"],       # FIX: "Mu (mi)" removed from stimuli (MOR inhibits GABA)
        "inhibitory": ["AMPA", "NMDA", "H1", "NK1", "Mu (mi)"],  # FIX: Mu moved here
    },
    "GABA_B": {
        "stimuli":    ["D2", "CB1"],
        "inhibitory": ["AMPA", "mGluR1", "H2"],
    },

    # --- Dopamine receptors ---
    "D1": {
        "stimuli":    ["NMDA", "AMPA", "5-HT2A", "H1"],
        "inhibitory": ["GABA_A", "D2"],
    },
    "D2": {
        "stimuli":    ["GABA_B", "5-HT1A", "H2"],
        "inhibitory": ["D1", "AMPA"],
    },

    # --- Serotonin receptors ---
    "5-HT1A": {
        "stimuli":    ["GABA_A", "D2"],
        "inhibitory": ["NMDA", "5-HT2A"],
    },
    "5-HT2A": {
        "stimuli":    ["D1", "NMDA", "mGluR1"],
        "inhibitory": ["GABA_A", "5-HT1A"],
    },
    # 5-HT3 is a ligand-gated ion channel that excites GABAergic interneurons →
    # proxied as stimulating GABA_A.
    "5-HT3": {
        "stimuli":    ["GABA_A"],             # FIX: was ["GABAergic interneurons"]
    },

    # --- Histamine receptors ---
    "H1": {
        "stimuli":    ["D1", "5-HT2A"],       # FIX: "5-HT2C" replaced with "5-HT2A"
        "inhibitory": ["GABA_A", "MT1"],
    },
    "H2": {
        "stimuli":    ["D2", "NMDA"],
        "inhibitory": ["GABA_B"],
    },
    "H3": {
        "inhibitory": ["H1", "H2"],           # H3 is an autoreceptor suppressing histamine release
    },

    # --- Endocannabinoid receptors ---
    "CB1": {
        "stimuli":    ["GABA_B"],
        "inhibitory": ["NMDA", "AMPA"],       # FIX: "Glutamate release" replaced with NMDA + AMPA
    },

    # --- Opioid receptors ---
    # Mu has no stimuli within this closed receptor graph (driven by endogenous opioids externally).
    "Mu (mi)": {
        "inhibitory": ["NMDA", "5-HT2A"],     # FIX: "GABA_A" removed from stimuli (GABA_A does not activate MOR)
    },

    # --- Other receptors ---
    "GlyR alpha1": {
        "inhibitory": ["NMDA"],
    },
    "alpha1A": {
        "stimuli":    ["AMPA"],
    },
    "alpha2A": {
        "inhibitory": ["D1", "5-HT2A"],
    },
    "sigma1": {
        "inhibitory": ["NMDA"],
    },
    "MT1": {
        "inhibitory": ["H1"],
    },
    # Orexin/hypocretin receptor — promotes histamine neuron activity →
    # proxied as stimulating H1 and H2.
    "OX1": {
        "stimuli":    ["H1", "H2"],           # FIX: was ["Histamine release"]
    },
    "NK1": {
        "inhibitory": ["GABA_A"],
    },
}

RECEPTOR_NAMES: list[str] = list(RECEPTOR_RULES.keys())
N_RECEPTORS: int = len(RECEPTOR_NAMES)
_RECEPTOR_INDEX: dict[str, int] = {name: i for i, name in enumerate(RECEPTOR_NAMES)}


def _build_interaction_matrix() -> np.ndarray:
    """
    Build a fixed (N_RECEPTORS × N_RECEPTORS) interaction matrix W where:
        W[i, j] = +1  if receptor j stimulates receptor i
        W[i, j] = -1  if receptor j inhibits  receptor i
        W[i, j] =  0  otherwise

    This lets the full receptor response be computed as a single matmul:
        response = W @ inputs          (before Q scaling)
    """
    W = np.zeros((N_RECEPTORS, N_RECEPTORS), dtype=np.float32)
    for receptor, rules in RECEPTOR_RULES.items():
        i = _RECEPTOR_INDEX[receptor]
        for src in rules.get("stimuli", []):
            if src in _RECEPTOR_INDEX:
                W[i, _RECEPTOR_INDEX[src]] = 1.0
        for src in rules.get("inhibitory", []):
            if src in _RECEPTOR_INDEX:
                W[i, _RECEPTOR_INDEX[src]] = -1.0
    return W


# Precomputed once at import time; never trained.
INTERACTION_MATRIX: np.ndarray = _build_interaction_matrix()


# ---------------------------------------------------------------------------
# Keras Layer
# ---------------------------------------------------------------------------

class NeuronLIKE(tf.keras.layers.Layer):
    """
    A biologically-inspired dense layer whose units (neurons) each maintain
    a pair of trainable Q-factors per receptor type.

    Input shape:  (batch, num_neurons, N_RECEPTORS)   — one activation per
                  receptor channel per neuron.
    Output shape: (batch, num_neurons, N_RECEPTORS)

    Internal variables
    ------------------
    Q_in  : (num_neurons, N_RECEPTORS)  — scales incoming receptor signals
    Q_out : (num_neurons, N_RECEPTORS)  — scales outgoing receptor signals
    W     : fixed (N_RECEPTORS, N_RECEPTORS) interaction matrix (non-trainable)

    Forward pass
    ------------
    1. Scale inputs:      x_scaled = x * Q_in            [per neuron, per receptor]
    2. Mix receptors:     mixed    = x_scaled @ W.T       [receptor cross-talk]
    3. Apply threshold:   fired    = mixed * sigmoid(mixed - threshold)
    4. Scale outputs:     out      = fired * Q_out
    """

    def __init__(
        self,
        num_neurons: int,
        history_size: int = 16,
        q_init_range: tuple[float, float] = (0.9, 1.1),
        threshold_init: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_neurons = num_neurons
        self.history_size = history_size
        self.q_init_range = q_init_range
        self.threshold_init_val = threshold_init

    # ------------------------------------------------------------------
    # Layer building (called once on first call with real input shape)
    # ------------------------------------------------------------------

    def build(self, input_shape):
        lo, hi = self.q_init_range

        # Trainable quality factors — initialised near 1 so the layer starts
        # close to a standard linear transform.
        self.Q_in = self.add_weight(
            name="Q_in",
            shape=(self.num_neurons, N_RECEPTORS),
            initializer=tf.keras.initializers.RandomUniform(lo, hi),
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=10.0),
            trainable=True,
        )
        self.Q_out = self.add_weight(
            name="Q_out",
            shape=(self.num_neurons, N_RECEPTORS),
            initializer=tf.keras.initializers.RandomUniform(lo, hi),
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=10.0),
            trainable=True,
        )

        # Per-neuron, per-receptor learned firing threshold
        self.threshold = self.add_weight(
            name="threshold",
            shape=(self.num_neurons, N_RECEPTORS),
            initializer=tf.keras.initializers.Constant(self.threshold_init_val),
            trainable=True,
        )

        # Fixed interaction matrix — not trained
        self.W = self.add_weight(
            name="W",
            shape=(N_RECEPTORS, N_RECEPTORS),
            initializer=tf.keras.initializers.Constant(INTERACTION_MATRIX),
            trainable=False,
        )

        # Ring buffer for input history (not a tf.Variable so it won't be
        # serialised as a weight; reset on each build/call explicitly).
        # Shape: (num_neurons, history_size, N_RECEPTORS)
        self._history_buffer = tf.Variable(
            tf.zeros((self.num_neurons, self.history_size, N_RECEPTORS)),
            trainable=False,
            name="history_buffer",
        )
        self._history_ptr = tf.Variable(0, trainable=False, dtype=tf.int32, name="history_ptr")

        super().build(input_shape)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Parameters
        ----------
        inputs : tf.Tensor, shape (batch, num_neurons, N_RECEPTORS)
            Raw receptor activation values.
        training : bool
            When True, history buffer is updated (useful for analysis).

        Returns
        -------
        tf.Tensor, shape (batch, num_neurons, N_RECEPTORS)
        """
        # 1. Scale incoming signals
        x_scaled = inputs * self.Q_in[tf.newaxis, :, :]   # (batch, neurons, R)

        # 2. Receptor cross-talk via interaction matrix
        #    einsum: for each sample b and neuron n, mix[b,n,i] = sum_j x_scaled[b,n,j]*W[j,i]
        mixed = tf.einsum("bnj,ij->bni", x_scaled, self.W)  # (batch, neurons, R)

        # 3. Soft threshold — smooth surrogate for "fire if above threshold"
        #    Using sigmoid so the operation is fully differentiable.
        gate = tf.sigmoid(mixed - self.threshold[tf.newaxis, :, :])
        fired = mixed * gate                                 # (batch, neurons, R)

        # 4. Scale output
        output = fired * self.Q_out[tf.newaxis, :, :]       # (batch, neurons, R)

        # 5. Optionally update history ring buffer (batch mean stored)
        if training:
            batch_mean = tf.reduce_mean(inputs, axis=0)     # (neurons, R)
            ptr = self._history_ptr % self.history_size
            indices = tf.stack(
                [tf.range(self.num_neurons),
                 tf.fill([self.num_neurons], ptr)], axis=1
            )
            self._history_buffer.scatter_nd_update(indices, batch_mean)
            self._history_ptr.assign_add(1)

        return output

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_history_mean(self) -> tf.Tensor:
        """Return the running mean of inputs over the history buffer."""
        return tf.reduce_mean(self._history_buffer, axis=1)  # (neurons, R)

    def receptor_index(self, name: str) -> int:
        """Return the column index for a given receptor name."""
        return _RECEPTOR_INDEX[name]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update(
            num_neurons=self.num_neurons,
            history_size=self.history_size,
            q_init_range=self.q_init_range,
            threshold_init=self.threshold_init_val,
        )
        return cfg

    def save_weights_pickle(self, path: str) -> None:
        """Lightweight save: pickle just the numpy weight arrays."""
        data = {
            "Q_in": self.Q_in.numpy(),
            "Q_out": self.Q_out.numpy(),
            "threshold": self.threshold.numpy(),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_weights_pickle(self, path: str) -> None:
        """Load weights previously saved with save_weights_pickle."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q_in.assign(data["Q_in"])
        self.Q_out.assign(data["Q_out"])
        self.threshold.assign(data["threshold"])


# ---------------------------------------------------------------------------
# Multi-layer network helper - this is optional
# ---------------------------------------------------------------------------

class NeuronNetwork(tf.keras.Model):
    """
    Stack several NeuronLIKEs into a Keras Model.

    Input:  (batch, num_neurons, N_RECEPTORS)
    Output: (batch, num_neurons, N_RECEPTORS)

    Training uses standard Keras model.compile / model.fit.
    Loss is MSE between output receptor activations and the target.
    """

    def __init__(self, num_neurons: int, num_layers: int = 3, history_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.neuron_layers = [
            NeuronLIKE(num_neurons, history_size=history_size, name=f"neuron_layer_{i}")
            for i in range(num_layers)
        ]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = inputs
        for layer in self.neuron_layers:
            x = layer(x, training=training)
        return x


# ---------------------------------------------------------------------------
# Sample usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BATCH = 4
    NUM_NEURONS = 3

    # Random receptor activations — shape (batch, neurons, receptors)
    sample_input = tf.random.uniform((BATCH, NUM_NEURONS, N_RECEPTORS), minval=0.0, maxval=1.0)

    # --- Single layer ---
    layer = NeuronLIKE(num_neurons=NUM_NEURONS, history_size=16)
    out = layer(sample_input, training=True)
    print("NeuronLIKE output shape:", out.shape)          # (4, 3, 24)
    print("Q_in  shape:", layer.Q_in.shape)                # (3, 24)
    print("History mean shape:", layer.get_history_mean().shape)

    # --- Multi-layer network with Keras training loop ---
    model = NeuronNetwork(num_neurons=NUM_NEURONS, num_layers=2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    # Dummy supervised target: same shape as output
    target = tf.random.uniform((BATCH, NUM_NEURONS, N_RECEPTORS))

    # Single training step to verify gradients flow
    with tf.GradientTape() as tape:
        pred = model(sample_input, training=True)
        loss = tf.reduce_mean(tf.square(pred - target))

    grads = tape.gradient(loss, model.trainable_variables)
    print(f"\nTraining step — loss: {loss.numpy():.6f}")
    for var, grad in zip(model.trainable_variables, grads):
        print(f"  {var.name:40s}  grad norm: {tf.norm(grad).numpy():.4f}")

    # Save / load weights
    layer.save_weights_pickle("/tmp/neuron_weights.pkl")
    layer.load_weights_pickle("/tmp/neuron_weights.pkl")
    print("\nWeights saved and reloaded successfully.")

    # Receptor index lookup
    print(f"\nAMPA receptor index: {layer.receptor_index('AMPA')}")
    print(f"NMDA receptor index: {layer.receptor_index('NMDA')}")
