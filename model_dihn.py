# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.contrib import layers
from tools import *

linear_parent_scope = "linear"
dnn_parent_scope = "dnn"


def model_fn(features, label_dict, fc_generator, is_training, keep_prob, params):
    # parse params
    black_list = params['black_list'] if 'black_list' in params else ""
    num_heads = params['num_heads'] if 'num_heads' in params else 2
    linear_key_dim = params['linear_key_dim'] if 'linear_key_dim' in params else 56
    linear_value_dim = params['linear_value_dim'] if 'linear_value_dim' in params else 56
    output_dim = params['output_dim'] if 'output_dim' in params else 56
    hidden_dim = params['hidden_dim'] if 'hidden_dim' in params else 112
    num_layer = params['num_layer'] if 'num_layer' in params else 2

    ########################################################
    # feature generating
    tf.logging.info("building features...")
    outputs_dict = fc_generator.get_output_dict(features, black_list)

    tf.logging.info("finished build features:")
    for key in outputs_dict:
        tf.logging.info(key)
        tf.logging.info(outputs_dict[key])

    tf.logging.info("building user field features:")
    user_feats = []
    for key in outputs_dict:
        if "is_user" in key:
            tf.logging.info(key)
            user_feats.append((key, outputs_dict[key]))

    user_feats = [feat for _, feat in sorted(user_feats, key=lambda x: x[0])]

    tf.logging.info("building item field features:")
    item_feats = []
    for key in outputs_dict:
        if "is_item" in key:
            tf.logging.info(key)
            item_feats.append((key, outputs_dict[key]))

    item_feats = [feat for _, feat in sorted(item_feats, key=lambda x: x[0])]

    tf.logging.info("building trigger field features:")
    trigger_feats = []
    for key in outputs_dict:
        if "is_trigger" in key:
            tf.logging.info(key)
            trigger_feats.append((key, outputs_dict[key]))

    trigger_feats = [feat for _, feat in sorted(trigger_feats, key=lambda x: x[0])]

    tf.logging.info("building context field features:")
    context_feats = []
    for key in outputs_dict:
        if "is_context" in key:
            tf.logging.info(key)
            context_feats.append((key, outputs_dict[key]))

    context_feats = [feat for _, feat in sorted(context_feats, key=lambda x: x[0])]

    # click behaviors
    shared_click_city_id = outputs_dict["shared_click_city_id"]
    shared_click_city_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_click_city_id], axis=1)

    shared_click_cate_id = outputs_dict["shared_click_cate_id"]
    shared_click_cate_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_click_cate_id], axis=1)

    shared_click_type_id = outputs_dict["shared_click_type_id"]
    shared_click_type_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_click_type_id], axis=1)

    shared_click_item_id = outputs_dict["shared_click_item_id"]
    shared_click_item_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_click_item_id], axis=1)

    shared_click_poi_id = outputs_dict["shared_click_poi_id"]
    shared_click_poi_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_click_poi_id], axis=1)

    shared_click_tag_id = outputs_dict["shared_click_tag_id"]
    shared_click_tag_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_click_tag_id], axis=1)

    clicks_set_click_time = outputs_dict["clicks_set_click_time"]
    clicks_set_click_time = tf.concat(clicks_set_click_time, axis=1)

    clicks_set_clicks_set_mask = outputs_dict["clicks_set_clicks_set_mask"]
    clicks_set_clicks_set_mask = tf.concat(clicks_set_clicks_set_mask, axis=1)

    # subsequence----subcate click behaviors
    shared_subcate_click_city_id = outputs_dict["shared_subcate_click_city_id"]
    shared_subcate_click_city_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_subcate_click_city_id],
                                             axis=1)

    shared_subcate_click_cate_id = outputs_dict["shared_subcate_click_cate_id"]
    shared_subcate_click_cate_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_subcate_click_cate_id],
                                             axis=1)

    shared_subcate_click_type_id = outputs_dict["shared_subcate_click_type_id"]
    shared_subcate_click_type_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_subcate_click_type_id],
                                             axis=1)

    shared_subcate_click_item_id = outputs_dict["shared_subcate_click_item_id"]
    shared_subcate_click_item_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_subcate_click_item_id],
                                             axis=1)

    shared_subcate_click_poi_id = outputs_dict["shared_subcate_click_poi_id"]
    shared_subcate_click_poi_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_subcate_click_poi_id], axis=1)

    shared_subcate_click_tag_id = outputs_dict["shared_subcate_click_tag_id"]
    shared_subcate_click_tag_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_subcate_click_tag_id], axis=1)


    # subsequence----subdest click behaviors
    shared_subdest_click_city_id = outputs_dict["shared_subdest_click_city_id"]
    shared_subdest_click_city_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_subdest_click_city_id],
                                             axis=1)

    shared_subdest_click_cate_id = outputs_dict["shared_subdest_click_cate_id"]
    shared_subdest_click_cate_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_subdest_click_cate_id],
                                             axis=1)

    shared_subdest_click_type_id = outputs_dict["shared_subdest_click_type_id"]
    shared_subdest_click_type_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_subdest_click_type_id],
                                             axis=1)

    shared_subdest_click_item_id = outputs_dict["shared_subdest_click_item_id"]
    shared_subdest_click_item_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_subdest_click_item_id],
                                             axis=1)

    shared_subdest_click_poi_id = outputs_dict["shared_subdest_click_poi_id"]
    shared_subdest_click_poi_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_subdest_click_poi_id], axis=1)

    shared_subdest_click_tag_id = outputs_dict["shared_subdest_click_tag_id"]
    shared_subdest_click_tag_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_subdest_click_tag_id], axis=1)


    # target item
    item_id = outputs_dict["is_item_f1"]
    cate_id = outputs_dict["is_item_f2"]
    city_id = outputs_dict["is_item_f3"]
    poi_id = outputs_dict["is_item_f4"]
    tag_id = outputs_dict["is_item_f5"]

    query_target = tf.concat([item_id, city_id, cate_id, poi_id, tag_id], axis=1)

    # trigger item
    t_item_id = outputs_dict["is_trigger_f1"]
    t_cate_id = outputs_dict["is_trigger_f2"]
    t_city_id = outputs_dict["is_trigger_f3"]
    t_poi_id = outputs_dict["is_trigger_f4"]
    t_tag_id = outputs_dict["is_trigger_f5"]

    query_trigger = tf.concat([t_item_id, t_city_id, t_cate_id, t_poi_id, t_tag_id], axis=1)

    activation_fn = tf.nn.relu
    # used for mixture tensor
    activation_fn_2 = tf.nn.sigmoid 

    ########################################################
    # User Intent Network
    with tf.variable_scope("user_intent_net"):
        clicks_features = tf.concat(
            [shared_click_item_id, shared_click_city_id, shared_click_cate_id, shared_click_type_id,
             shared_click_poi_id, shared_click_tag_id], axis=2)

        clicks_pool_res = time_attention_pooling(clicks_features, query_trigger, clicks_set_click_time,
                                                 clicks_set_clicks_set_mask, False, 'click_attention_pooling')

        uin_raw_fea = user_feats + trigger_feats + [clicks_pool_res]
        input = tf.concat(uin_raw_fea, axis=1)
        input = layers.fully_connected(input, 256, activation_fn=None, scope='ffn_1',
                                       variables_collections=[dnn_parent_scope])

        input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
                                  variables_collections=[dnn_parent_scope])

        fusing_gate = layers.fully_connected(input, 40, activation_fn=activation_fn_2, scope='ffn_2',
                                       variables_collections=[dnn_parent_scope])

        input = layers.batch_norm(fusing_gate, is_training=is_training, activation_fn=activation_fn,
                                  variables_collections=[dnn_parent_scope])

        uin_logit = layers.fully_connected(input, 1, activation_fn=None, scope='ffn_4',
                                       variables_collections=[dnn_parent_scope])


    #fusing embedding moudle
    fusing_embedding = tf.multiply(fusing_gate,query_trigger) + tf.multiply(1-fusing_gate,query_target)


    ########################################################
    # hybrid interest extracting module
    with tf.variable_scope("hybrid_interest_extract"):
        with tf.variable_scope("hard_seq_modeling_subcate"):
            subcate_clicks_features = tf.concat(
                [shared_subcate_click_item_id, shared_subcate_click_city_id, shared_subcate_click_cate_id, shared_subcate_click_type_id,
                shared_subcate_click_poi_id, shared_subcate_click_tag_id], axis=2)
            clicks_trans_block = SelfAttentionPooling(
                num_heads=num_heads,
                key_mask=clicks_set_clicks_set_mask,
                query_mask=clicks_set_clicks_set_mask,
                length=30,
                linear_key_dim=linear_key_dim,
                linear_value_dim=linear_value_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_layer=num_layer,
                keep_prob=keep_prob
        )
            clicks_trans_output = clicks_trans_block.build(subcate_clicks_features, reuse=False,
                                                       scope='clicks_trans')  # (batch_size, 30, output_dim)
            subcate_clicks_pool_res = tf.reduce_mean(clicks_trans_output, axis=1)

        with tf.variable_scope("hard_seq_modeling_sudest"):
            subdest_clicks_features = tf.concat(
                [shared_subdest_click_item_id, shared_subdest_click_city_id, shared_subdest_click_cate_id, shared_subdest_click_type_id,
                shared_subdest_click_poi_id, shared_subdest_click_tag_id], axis=2)
            clicks_trans_block = SelfAttentionPooling(
                num_heads=num_heads,
                key_mask=clicks_set_clicks_set_mask,
                query_mask=clicks_set_clicks_set_mask,
                length=30,
                linear_key_dim=linear_key_dim,
                linear_value_dim=linear_value_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_layer=num_layer,
                keep_prob=keep_prob
        )
            clicks_trans_output = clicks_trans_block.build(subdest_clicks_features, reuse=False,
                                                       scope='clicks_trans')  # (batch_size, 30, output_dim)
            subdest_clicks_pool_res = tf.reduce_mean(clicks_trans_output, axis=1)

        with tf.variable_scope("soft_seq_modeling"):
            clicks_trans_block = SelfAttentionPooling(
                num_heads=num_heads,
                key_mask=clicks_set_clicks_set_mask,
                query_mask=clicks_set_clicks_set_mask,
                length=30,
                linear_key_dim=linear_key_dim,
                linear_value_dim=linear_value_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_layer=num_layer,
                keep_prob=keep_prob
            )
            clicks_trans_output = clicks_trans_block.build(clicks_features, reuse=False,
                                                           scope='clicks_trans')  # (batch_size, 30, output_dim)

            main_clicks_pool_res = time_attention_pooling(clicks_trans_output, fusing_embedding, clicks_set_click_time,
                                                     clicks_set_clicks_set_mask, False, 'click_attention_pooling')


    ########################################################
    # logits
    with tf.variable_scope('logit'):
        input = user_feats + item_feats + context_feats + trigger_feats + [subcate_clicks_pool_res] + [subdest_clicks_pool_res] + [main_clicks_pool_res]
        input = tf.concat(input, axis=1)
        print(input)
        input = layers.batch_norm(input, is_training=is_training, activation_fn=None,
                                  variables_collections=[dnn_parent_scope])

        input = layers.fully_connected(input, 512, activation_fn=None, scope='ffn_1',
                                       variables_collections=[dnn_parent_scope])

        input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
                                  variables_collections=[dnn_parent_scope])

        input = layers.fully_connected(input, 256, activation_fn=None, scope='ffn_2',
                                       variables_collections=[dnn_parent_scope])

        input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
                                  variables_collections=[dnn_parent_scope])

        input = layers.fully_connected(input, 128, activation_fn=None, scope='ffn_3',
                                       variables_collections=[dnn_parent_scope])

        input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
                                  variables_collections=[dnn_parent_scope])

        logit = layers.fully_connected(input, 1, activation_fn=None, scope='ffn_4',
                                       variables_collections=[dnn_parent_scope])
        logit = logit

    logit_dict = {}
    logit_dict['ctr'] = logit
    logit_dict['aux_ctr'] = uin_logit

    label_click = label_dict['click']
    label_click = tf.cast(tf.equal(label_click, '1'), tf.float32)
    label_click = tf.reshape(label_click, [-1, 1])
    label_dict['click'] = label_click
    trigger_click = label_dict['trigger_click']
    trigger_click = tf.cast(tf.equal(trigger_click, '1'), tf.float32)
    trigger_click = tf.reshape(trigger_click, [-1, 1])
    label_dict['trigger_click'] = trigger_click

    return logit_dict, label_dict


class SelfAttentionPooling:
    """SelfAttentionPooling class"""

    def __init__(self,
                 num_heads,
                 key_mask,
                 query_mask,
                 length,
                 linear_key_dim,
                 linear_value_dim,
                 output_dim,
                 hidden_dim,
                 num_layer,
                 keep_prob):
        """
        :param key_mask: mask matrix for key
        :param query_mask: mask matrix for query
        :param num_heads: number of multi-attention head
        :param linear_key_dim: key, query forward dim
        :param linear_value_dim: val forward dim
        :param output_dim: fnn output dim
        :param hidden_dim: fnn hidden dim
        :param num_layer: number of multi-attention layer
        :param keep_prob: keep probability
        """
        assert linear_key_dim % num_heads == 0
        assert linear_value_dim % num_heads == 0

        self.length = length
        self.num_layers = num_layer
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.key_mask = key_mask
        self.query_mask = query_mask
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.keep_prob = keep_prob

    def build(self, inputs, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            # o1 = self._positional_add(inputs)
            o1 = inputs
            for i in range(1, self.num_layers + 1):
                with tf.variable_scope("layer_{}".format(i)):
                    o1_ = self.multi_head(o1, o1, o1, 'multi_head')
                    o2 = self._add_and_norm(o1, o1_, 'norm_1')
                    o2_ = self._positional_feed_forward(o2, self.hidden_dim, self.output_dim, 'forward')
                    o3 = self._add_and_norm(o2, o2_, 'norm_2')
                    o1 = o3
            return o1

    def _positional_feed_forward(self, output, hidden_dim, output_dim, scope):
        with tf.variable_scope(scope):
            output = layers.fully_connected(output, hidden_dim, activation_fn=tf.nn.relu,
                                            variables_collections=[dnn_parent_scope])
            output = layers.fully_connected(output, output_dim, activation_fn=None,
                                            variables_collections=[dnn_parent_scope])
            return tf.nn.dropout(output, self.keep_prob)

    def _add_and_norm(self, x, sub_layer_x, scope):
        with tf.variable_scope(scope):
            return layers.layer_norm(tf.add(x, sub_layer_x), variables_collections=[dnn_parent_scope])

    def multi_head(self, q, k, v, scope):
        with tf.variable_scope(scope):
            q, k, v = self._linear_projection(q, k, v)
            qs, ks, vs = self._split_heads(q, k, v)
            outputs = self._scaled_dot_product(qs, ks, vs)
            output = self._concat_heads(outputs)
            return tf.nn.dropout(output, self.keep_prob)

    def _linear_projection(self, q, k, v):
        q = layers.fully_connected(q, self.linear_key_dim, biases_initializer=None, activation_fn=None,
                                   variables_collections=[dnn_parent_scope])
        k = layers.fully_connected(k, self.linear_key_dim, biases_initializer=None, activation_fn=None,
                                   variables_collections=[dnn_parent_scope])
        v = layers.fully_connected(v, self.linear_value_dim, biases_initializer=None, activation_fn=None,
                                   variables_collections=[dnn_parent_scope])
        return q, k, v

    def _split_heads(self, q, k, v):
        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, dim]

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.linear_key_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.linear_key_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.linear_value_dim)
        return qs, ks, vs

    def _scaled_dot_product(self, qs, ks, vs):
        key_dim_per_head = self.linear_key_dim // self.num_heads

        o1 = tf.matmul(qs, ks, transpose_b=True)
        # o1 = tf.matmul(layers.layer_norm(qs, variables_collections=[dnn_parent_scope]),
        #                layers.layer_norm(ks, variables_collections=[dnn_parent_scope]), transpose_b=True)
        o2 = o1 / (key_dim_per_head ** 0.5)  # (batch_size, num_heads, q_length, k_length)
        if self.key_mask is not None:  # (batch_size, k_length)
            # key mask
            padding_num = -2 ** 32 + 1
            # Generate masks
            mask = tf.expand_dims(self.key_mask, 1)  # (batch_size, 1, k_length)
            mask = tf.tile(mask, [1, self.length, 1])  # (batch_size, q_length, k_length)
            mask = tf.expand_dims(mask, 1)
            mask = tf.tile(mask, [1, self.num_heads, 1, 1])
            # Apply masks to inputs
            paddings = tf.ones_like(o2) * padding_num
            o2 = tf.where(tf.equal(mask, 0), paddings, o2)  # (batch_size, num_heads, q_length, k_length)
        o3 = tf.nn.softmax(o2)

        if self.query_mask is not None:
            mask = tf.expand_dims(self.query_mask, 2)  # (batch_size, q_length, 1)
            mask = tf.expand_dims(mask, 1)
            mask = tf.tile(mask, [1, self.num_heads, 1, 1])  # (batch_size, num_heads, q_length, 1)
            o3 = o3 * tf.cast(mask, tf.float32)  # broadcast
        return tf.matmul(o3, vs)

    def _concat_heads(self, outputs):
        def transpose_then_concat_last_two_dimenstion(tensor):
            tensor = tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, max_seq_len, num_heads, dim]
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

        return transpose_then_concat_last_two_dimenstion(outputs)



def time_attention_pooling(inputs, memory, time, mask, reuse, scope='pool_attention'):
    with tf.variable_scope(scope, reuse=reuse):
        _, JX, dim = inputs.get_shape().as_list()
        with tf.variable_scope("time"):
            time = tf.expand_dims(tf.log(time + 2), axis=2)
            # time = tf.expand_dims(time, axis=2)
            time = layers.fully_connected(time, 8, activation_fn=tf.nn.tanh, scope='att_time',
                                          variables_collections=[dnn_parent_scope])
        memory = tf.tile(tf.expand_dims(memory, axis=1), [1, JX, 1])
        with tf.variable_scope("attention"):
            u = tf.concat([memory, inputs, time], axis=2)
            u = layers.fully_connected(u, 64, activation_fn=tf.nn.relu, scope='att_dense1',
                                       variables_collections=[dnn_parent_scope])
            u = layers.fully_connected(u, 32, activation_fn=tf.nn.relu, scope='att_dense2',
                                       variables_collections=[dnn_parent_scope])
            s = layers.fully_connected(u, 1, activation_fn=None, scope='att_dense3',
                                       variables_collections=[dnn_parent_scope])
            if mask is not None:
                s = softmax_mask(tf.squeeze(s, [2]), mask)
                a = tf.expand_dims(tf.nn.softmax(s), axis=2)
            else:
                a = tf.expand_dims(tf.nn.softmax(tf.squeeze(s, [2])), axis=2)
            res = tf.squeeze(tf.matmul(inputs, a, transpose_a=True), axis=2)
            return res
