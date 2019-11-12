import sys
import string

import numpy as np
import tensorflow as tf

class Config:
    def __init__(self):
        self.name = "att-bilstm-cnn"
        
        # Word
        self.w_embedding_d = 300
        self.w_feature_d = 4
        
        # Character
        self.c_embedding_d = 25
        self.c_feature_d = 4
        self.w_max_length = 20
        self.cnn_window_to_kernels = {1: 20, 2: 20, 3: 20}
        
        # BiLSTM
        self.is_cross_bilstm = False
        self.hidden_d = 100
        self.layers = 2
        
        # Attention
        self.attention_heads = 5
        
        # Classifier
        self.output_d = 5
        
        # Optimization
        self.max_gradient_norm = 0
        self.learning_rate = 1e-3
        self.keep_rate = 0.65
        return
        
class CNN:
    def __init__(self, name="cnn", input_d=5, window_to_kernels={2:3}):
        with tf.variable_scope(name):
            self.W_list = []
            for window, kernels in window_to_kernels.items():
                self.W_list.append(
                    tf.get_variable(f"w{window}", [1, window, input_d, kernels])
                )
        return
        
    def transform(self, X):
        """
        X: [batch, length, dimension]
        H: [batch, total_kernels]
        """
        X = tf.expand_dims(X, 1) # [batch, 1, length, dimension]
        
        H_list = []
        for W in self.W_list:
            H = tf.nn.conv2d(X, W, [1,1,1,1], "VALID") # [batch, 1, length-window+1, kernels]
            H = tf.reduce_max(H, [1,2]) # [batch, kernels]
            H_list.append(H)
            
        H = tf.concat(H_list, 1) # [batch, total_kernels]
        return H
        
class BiLSTM:
    def __init__(self, name="bilstm", input_d=300, hidden_d=100, layers=2):
        self.name = name
        self.kr = tf.placeholder(tf.float32, [])
        
        f_cell_list = []
        b_cell_list = []
        for layer in range(layers):
            is_top_cell = (layer==layers-1)
            current_input_d = input_d if layer==0 else hidden_d
            f_cell_list.append(
                self.create_cell(current_input_d, hidden_d, is_top_cell)
            )
            b_cell_list.append(
                self.create_cell(current_input_d, hidden_d, is_top_cell)
            )
        self.f_cell = tf.nn.rnn_cell.MultiRNNCell(f_cell_list)
        self.b_cell = tf.nn.rnn_cell.MultiRNNCell(b_cell_list)
        return
        
    def create_cell(self, input_d, hidden_d, is_top_cell):
        cell = tf.nn.rnn_cell.LSTMCell(hidden_d)
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob = self.kr,
            state_keep_prob = self.kr,
            output_keep_prob = self.kr if is_top_cell else 1.0,
            variational_recurrent = True,
            input_size = input_d,
            dtype = tf.float32,
        )
        return cell
        
    def transform(self, X, L):
        """BiLSTM transform function.
        
        X: [batch, length, input_d]
        L: [batch]
        H: [batch, length, hidden_d*2]
        """
        with tf.variable_scope(self.name):
            # top_output: [2, batch, length, hidden_d], axis0: forward/backward
            # last_state: [2, layers, 2, batch, hidden_d], axis0: forward/backward, axis2: LSTM c/h
            top_output, last_state = tf.nn.bidirectional_dynamic_rnn(
                self.f_cell,
                self.b_cell,
                X,
                sequence_length = L,
                dtype = tf.float32,
            )
        H = tf.concat(top_output, 2) # [batch, length, hidden_d*2]
        return H
        
class Cross_BiLSTM:
    def __init__(self, name="cross-bilstm", input_d=300, hidden_d=100, layers=2):
        self.name = name
        self.kr = tf.placeholder(tf.float32, [])
        
        self.f_cell_list = []
        self.b_cell_list = []
        for layer in range(layers):
            is_top_cell = (layer==layers-1)
            current_input_d = input_d if layer==0 else hidden_d*2
            self.f_cell_list.append(
                self.create_cell(current_input_d, hidden_d, is_top_cell, f"lstm_cell_{layer+1}")
            )
            self.b_cell_list.append(
                self.create_cell(current_input_d, hidden_d, is_top_cell, f"lstm_cell_{layer+1}")
            )
        return
        
    def create_cell(self, input_d, hidden_d, is_top_cell, name):
        cell = tf.nn.rnn_cell.LSTMCell(hidden_d, name=name)
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob = self.kr,
            state_keep_prob = self.kr,
            output_keep_prob = self.kr if is_top_cell else 1.0,
            variational_recurrent = True,
            input_size = input_d,
            dtype = tf.float32,
        )
        return cell
        
    def transform(self, X, L):
        """BiLSTM transform function.
        
        X: [batch, length, input_d]
        L: [batch]
        H: [batch, length, hidden_d*2]
        """
        H = X
        with tf.variable_scope(self.name):
            for f_cell, b_cell in zip(self.f_cell_list, self.b_cell_list):
                # top_output: [2, batch, length, hidden_d], axis0: forward/backward
                # last_state: [2, layers, 2, batch, hidden_d], axis0: forward/backward, axis2: LSTM c/h
                top_output, last_state = tf.nn.bidirectional_dynamic_rnn(
                    f_cell,
                    b_cell,
                    H,
                    sequence_length = L,
                    dtype = tf.float32,
                )
                H = tf.concat(top_output, 2) # [batch, length, hidden_d*2]
        return H
        
class Att:
    def __init__(self, name="att", heads=5, dk_head=60, dv_head=60):
        self.heads = heads
        self.QP_list = []
        self.KP_list = []
        self.VP_list = []
        for i in range(heads):
            self.QP_list.append(
                tf.layers.Dense(dk_head, use_bias=False, name=f"{name}/QP_{i}")
            )
            self.KP_list.append(
                tf.layers.Dense(dk_head, use_bias=False, name=f"{name}/KP_{i}")
            )
            self.VP_list.append(
                tf.layers.Dense(dv_head, use_bias=False, name=f"{name}/VP_{i}")
            )
        return
        
    def attention_transform(self, Q, K, V, L=None):
        """
        Q: [batch, q, dk]
        K: [batch, m, dk]
        V: [batch, m, dv]
        L: [batch]
        C: [batch, q, dv]
        """
        # Attention score
        dk = tf.cast(tf.shape(K)[2], tf.float32)
        score = tf.matmul(Q, K, transpose_b=True) # [batch, q, m]
        score = score/tf.sqrt(dk)                 # [batch, q, m]
        
        # Mask score
        if L is not None:
            mask = tf.sequence_mask(L)                                     # [batch, m]
            mask = tf.expand_dims(mask, 1)                                 # [batch, 1, m]
            mask = tf.tile(mask, [1, tf.shape(score)[1], 1])               # [batch, q, m]
            worst_score = tf.ones_like(score) * tf.constant(float("-inf")) # [batch, q, m]
            score = tf.where(mask, score, worst_score)                     # [batch, q, m]
            
        # Context vector
        alpha = tf.nn.softmax(score, 2) # [batch, q, m]
        C = tf.matmul(alpha, V)         # [batch, q, dv]
        return C
        
    def transform(self, Q, K, V, L=None):
        """
        Q: [batch, q, dq]
        K: [batch, m, dk]
        V: [batch, m, dv]
        C: [batch, q, heads * dv_head]
        """
        batch = tf.shape(Q)[0]
        q = tf.shape(Q)[1]
        m = tf.shape(K)[1]
        
        C_list = []
        for i in range(self.heads):
            Q_head = self.QP_list[i](Q) # [batch, q, dk_head]
            K_head = self.KP_list[i](K) # [batch, m, dk_head]
            V_head = self.VP_list[i](V) # [batch, m, dv_head]
            
            C_head = self.attention_transform(Q_head, K_head, V_head, L=L) # [batch, q, dv_head]
            C_list.append(C_head)
            
        C = tf.concat(C_list, 2) # [batch, q, heads * dv_head]
        return C
        
class Model:
    def __init__(self, config):
        """Contruct Tensowflow graph."""
        
        self.create_hyper_parameter(config)
        self.create_embedding()
        self.create_encoder()
        self.create_classifier()
        return
        
    def create_hyper_parameter(self, config):
        """Add attributes of config to self."""
        
        for parameter in dir(config):
            if parameter[0] == "_": continue
            setattr(self, parameter, getattr(config, parameter))
        return
        
    def create_embedding(self):
        """Create a trainable unknown word vector and trainable ascii character vectors."""
        
        with tf.variable_scope(self.name, initializer=tf.random_normal_initializer(stddev=0.1)):
            self.unknown_w_v = tf.get_variable("unknown_w_v", [1, self.w_embedding_d+self.w_feature_d])
            
        character_list = list(
            string.digits
          + string.ascii_lowercase
          + string.ascii_uppercase
        ) + [".", "-", ",", ")", "(", "\"", "'", ":", "/", "$", "+", ";", "*", "&", "=", "%", "?", "[", "]", "@", "!", "`", "#"]
        # character_list = list(string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation)
        character_list += ["UNK", "PAD"]
        self.c_token_to_index = {
            c: i
            for i, c in enumerate(character_list)
        }
        
        with tf.variable_scope(self.name, initializer=tf.random_normal_initializer(stddev=0.1)):
            self.c_embedding_table = tf.get_variable("c_embedding_table", [len(character_list), self.c_embedding_d])
        return
        
    def create_encoder(self):
        """Create an Att-BiLSTM-CNN encoder."""
        
        # Input sequence                              [batch, length]
        self.input_length = tf.placeholder(  tf.int32, [None])
        self.w_k          = tf.placeholder(  tf.int32, [None, None])
        self.w_v          = tf.placeholder(tf.float32, [None, None, self.w_embedding_d+self.w_feature_d])
        self.c_i          = tf.placeholder(  tf.int32, [None, None, self.w_max_length])
        self.c_f          = tf.placeholder(tf.float32, [None, None, self.w_max_length, self.c_feature_d])
        
        batch  = tf.shape(self.w_v)[0]
        length = tf.shape(self.w_v)[1]
        
        # Word
        w_known     = tf.equal(tf.reshape(self.w_k, [batch*length]), 1)                         # [batch * length]
        w_known_v   = tf.reshape(self.w_v, [batch*length, self.w_embedding_d+self.w_feature_d]) # [batch * length, w_embedding_d+w_feature_d]
        w_unknown_v = tf.tile(self.unknown_w_v, [batch*length, 1])                              # [batch * length, w_embedding_d+w_feature_d]
        w_v         = tf.where(w_known, w_known_v, w_unknown_v)                                 # [batch * length, w_embedding_d+w_feature_d]
        w_v         = tf.reshape(w_v, [batch, length, self.w_embedding_d+self.w_feature_d])     # [batch,  length, w_embedding_d+w_feature_d]
        
        # CNN
        cnn_input_d = self.c_embedding_d + self.c_feature_d
        cnn_output_d = sum(self.cnn_window_to_kernels.values())
        with tf.variable_scope(self.name):
            cnn = CNN(
                input_d = cnn_input_d,
                window_to_kernels = self.cnn_window_to_kernels,
            )
            c_v = tf.gather(self.c_embedding_table, self.c_i) # [batch, length, w_max_length, c_embedding_d]
            c_v = tf.concat([c_v, self.c_f], 3)               # [batch, length, w_max_length, c_embedding_d+c_feature_d]
            c_v = tf.reshape(c_v, [batch*length, self.w_max_length, cnn_input_d])
            c_based_w_v = cnn.transform(c_v)                  # [batch*length, cnn_output_d]
            c_based_w_v = tf.reshape(c_based_w_v, [batch, length, cnn_output_d])
            wc_v = tf.concat([w_v, c_based_w_v], 2)           # [batch, length, w_embedding_d+w_feature_d+cnn_output_d]
            
        # BiLSTM
        with tf.variable_scope(self.name):
            if self.is_cross_bilstm:
                self.bilstm = Cross_BiLSTM(
                    input_d = self.w_embedding_d + self.w_feature_d + cnn_output_d,
                    hidden_d = self.hidden_d,
                    layers = self.layers,
                )
            else:
                self.bilstm = BiLSTM(
                    input_d = self.w_embedding_d + self.w_feature_d + cnn_output_d,
                    hidden_d = self.hidden_d,
                    layers = self.layers,
                )
            w_h = self.bilstm.transform(wc_v, self.input_length) # [batch, length, hidden_d*2]
            
        # Multi-head attention
        if self.attention_heads == 0:
            self.w_a = w_h
        else:
            with tf.variable_scope(self.name):
                head_d = int(self.hidden_d * 2 / self.attention_heads)
                att = Att(
                    heads = self.attention_heads,
                    dk_head = head_d,
                    dv_head = head_d,
                )
                w_c = att.transform(w_h, w_h, w_h, L=self.input_length) # [batch, length, hidden_d*2]
            self.w_a = tf.concat([w_h, w_c], 2) # [batch, length, hidden_d*4]
        return
        
    def create_classifier(self):
        """Create a layer to compute predictions and loss."""
        
        # Output
        # self.logits are not actually logits, because a tf.nn.log_softmax() is omitted
        output_layer = tf.layers.Dense(
            self.output_d,
            use_bias = True,
            name = "output_layer",
        )
        with tf.variable_scope(self.name):
            self.logits = output_layer(self.w_a) # [batch, length, output_d]
            
        # Loss
        self.o_i    = tf.placeholder(  tf.int32, [None, None]) # [batch, length]
        self.o_mask = tf.placeholder(tf.float32, [None, None]) # [batch, length]
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = self.o_i,
            logits = self.logits,
        )
        self.loss = tf.reduce_sum(cross_entropy * self.o_mask) / tf.reduce_sum(self.o_mask)
        
        # Optimization
        optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=self.learning_rate)
        gv_list = optimizer.compute_gradients(self.loss)
        g_list, v_list = zip(*gv_list)
        g_list = [
            tf.convert_to_tensor(g) if isinstance(g, tf.IndexedSlices) else g
            for g in g_list
        ]
        if self.max_gradient_norm > 0:
            g_list, _ = tf.clip_by_global_norm(g_list, self.max_gradient_norm)
        self.update_op = optimizer.apply_gradients(zip(g_list, v_list))
        return
        
    def get_word_feature(self, word):
        """Get word captitalization pattern features."""
        
        feature_list = [0] * self.w_feature_d
        
        if word.isupper():
            feature_list[0] = 1
        
        elif word[0].isupper():
            feature_list[1] = 1
        
        elif word.islower():
            feature_list[2] = 1
        
        else:
            feature_list[3] = 1
            
        return feature_list
        
    def get_character_feature(self, character):
        feature_list = [0] * self.c_feature_d
        
        if len(character) != 1:
            return feature_list
            
        if character in string.ascii_uppercase:
            feature_list[0] = 1
            
        elif character in string.ascii_lowercase:
            feature_list[1] = 1
            
        elif character in string.digits:
            feature_list[2] = 1
            
        elif character in string.punctuation:
            feature_list[3] = 1
            
        return feature_list
        
    def get_word_for_cnn(self, word):
        word = [character for character in word[:self.w_max_length]]
        word += ["PAD"] * (self.w_max_length - len(word))
        return word
        
    def get_formatted_input(self, sample_list):
        """Preprocessing: extract Numpy arrays for TensorFlow input placeholders.
        
        sample_list: [sample]
            sample: (word_sequence, tokenlabel_sequence, chunk_to_entitytype)
        """
        batch = len(sample_list)
        length = max(len(word_sequence) for word_sequence, _, _ in sample_list)
        
        input_length = np.zeros([batch                                             ], dtype=np.int32)
        w_k          = np.zeros([batch, length                                     ], dtype=np.int32)
        w_v          = np.zeros([batch, length, self.w_embedding_d+self.w_feature_d], dtype=np.float32)
        c_i          = np.zeros([batch, length, self.w_max_length                  ], dtype=np.int32)
        c_f          = np.zeros([batch, length, self.w_max_length, self.c_feature_d], dtype=np.float32)
        o_i          = np.zeros([batch, length                                     ], dtype=np.int32)
        o_mask       = np.zeros([batch, length                                     ], dtype=np.float32)
        
        for b, (word_sequence, tokenlabel_sequence, _) in enumerate(sample_list):
            input_length[b] = len(word_sequence)
            
            for t, word in enumerate(word_sequence):
                if word in self.w_token_to_vector:
                    w_k[b,t] = 1
                    w_v[b,t] = np.concatenate((
                        self.w_token_to_vector[word],
                        self.get_word_feature(word),
                    ))
                    
                cnn_word = self.get_word_for_cnn(word)
                unk_index = self.c_token_to_index["UNK"]
                for c, character in enumerate(cnn_word):
                    c_i[b,t,c] = self.c_token_to_index.get(character, unk_index)
                    c_f[b,t,c] = self.get_character_feature(character)
                    
                o_i[b,t] = self.tokenlabel_to_index[tokenlabel_sequence[t]]
                o_mask[b,t] = 1
                
        return (
            input_length,
            w_k, w_v,
            c_i, c_f,
            o_i, o_mask,
        )
        
    def compute_loss_for_a_batch(self, sample_list, update_model=False):
        """Compute loss for a mini-batch, update model parameters if required.
        
        Return loss
            loss: the sum of cross-entropy across all tokens.
        """
        (
            input_length,
            w_k, w_v,
            c_i, c_f,
            o_i, o_mask,
        ) = self.get_formatted_input(sample_list)
        
        if update_model:
            output_list = [self.loss, self.update_op]
        else:
            output_list = [self.loss]
        
        result_list = self.sess.run(
            output_list,
            feed_dict = {
                self.bilstm.kr: self.keep_rate if update_model else 1.0,
                self.input_length: input_length,
                self.w_k: w_k,
                self.w_v: w_v,
                self.c_i: c_i,
                self.c_f: c_f,
                self.o_i: o_i,
                self.o_mask: o_mask,
            }
        )
        return result_list[0]
        
    def predict_label_for_a_batch(self, sample_list):
        """Generate token label sequence for each sample in the mini-batch.
        
        prediction_list: [prediction]
            prediction: [token_label]
        """
        (
            input_length,
            w_k, w_v,
            c_i, c_f,
            _, _,
        ) = self.get_formatted_input(sample_list)
        
        logits = self.sess.run( # [batch, length, output_d]
            self.logits,
            feed_dict = {
                self.bilstm.kr: 1.0,
                self.input_length: input_length,
                self.w_k: w_k,
                self.w_v: w_v,
                self.c_i: c_i,
                self.c_f: c_f,
            }
        )
        
        labelindex_sequence_list = np.argmax(logits, axis=2) # [batch, length]
        prediction_list = []
        
        for b, labelindex_sequence in enumerate(labelindex_sequence_list):
            prediction = []
            for label_index in labelindex_sequence[:input_length[b]]:
                token_label = self.tokenlabel_list[label_index]
                prediction.append(token_label)
            prediction_list.append(prediction)
            
        return prediction_list
        
def main():
    config = Config()
    model = Model(config)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.45
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        for v in tf.trainable_variables():
            print(v)
    return
    
if __name__ == "__main__":
    main()
    sys.exit()
    
