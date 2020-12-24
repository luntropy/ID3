#!/usr/bin/python3

from collections import deque

import random
import math

# Mapping
attr_by_pos = [ 'class', 'age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps', 'deg_malig', 'breast', 'breast_quad', 'irradiat' ]
pos_by_attr = { 'class': 0, 'age': 1, 'menopause': 2, 'tumor_size': 3, 'inv_nodes': 4, 'node_caps': 5, 'deg_malig': 6, 'breast': 7, 'breast_quad': 8, 'irradiat': 9 }
attr_values = [ [ 'no-recurrence-events', 'recurrence-events' ], [ '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99' ], [ 'lt40', 'ge40', 'premeno' ], [ '0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59' ], [ '0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36-39' ], [ 'yes', 'no' ], [ '1', '2', '3' ], [ 'left', 'right' ], [ 'left_up', 'left_low', 'right_up',	'right_low', 'central' ], [ 'yes', 'no' ] ]

CLASS_POSITION = 0

class Node:
    def __init__(self):
        self.parent = None
        self.attr = None
        self.value = None
        self.ft = {}
        self.values = []
        self.children = []

    def display(self, indent = 0):
        print(indent)
        if self.value == None:
            print(('\t' * indent) + self.attr)
        else:
            print(('\t' * indent) + self.value + ' -> ' + self.attr)

        for c in self.children:
            c.display(indent + 1)

    def get_child_by_value(self, value):
        for c in self.children:
            if c.value == value:
                return c

        return None

def create_dfs(data, lines_count, size_df, adjustments_cnt):
    df = []
    itr = 0

    while itr < lines_count:
        if adjustments_cnt > 0:
            temp_list = data[itr:itr+size_df+1]
            adjustments_cnt = adjustments_cnt - 1
            itr = itr + size_df + 1
        else:
            temp_list = data[itr:itr+size_df]
            itr = itr + size_df

        df.append(temp_list)

    return df

def separate_dfs(df, dfs_test, dfs_train):
    for frame in range(0, len(df)):
        dfs_test.append(df[frame])

        temp_df = []
        for i in range(0, len(df)):
            if i != frame:
                temp_df = temp_df + df[i]

        dfs_train.append(temp_df)

def entropy_one_attr(ft, S):
    entropy = 0

    for key in ft[S]:
        if key != 'total':
            if ft[S]['total'] != 0:
                prob = ft[S][key] / ft[S]['total']
            else:
                prob = 0

            if prob != 0:
                entropy = entropy - (prob * math.log(prob, 2))

    return entropy

def entropy_two_attr(ft, S):
    entropy = 0

    for key in ft[S]:
        if key != 'total':

            if ft[S]['total'] != 0:
                prob = ft[S][key]['total'] / ft[S]['total']
                ent = entropy_one_attr(ft[S], key)
            else:
                prob = 0

            entropy = entropy + (prob * ent)

    return entropy

def build_frequency_table(dfs_train, attr, ft):
    ft_attr = {}

    if attr == attr_by_pos[CLASS_POSITION]:
        for i in range(0, len(dfs_train)):
            example = dfs_train[i].split(',')

            if not ft_attr.keys():
                ft_attr = { example[CLASS_POSITION]: 1 }
            else:
                if example[CLASS_POSITION] not in ft_attr.keys():
                    ft_attr[example[CLASS_POSITION]] = 1
                else:
                    ft_attr[example[CLASS_POSITION]] = ft_attr[example[CLASS_POSITION]] + 1

        total = 0
        for key in ft_attr.keys():
            total = total + ft_attr[key]

        ft_attr['total'] = total
        ft[attr] = ft_attr

    else:
        for value in attr_values[pos_by_attr[attr]]:
            for given_class in attr_values[CLASS_POSITION]:
                if value not in ft_attr.keys():
                    ft_attr[value] = { given_class: 0 }
                else:
                    if not ft_attr[value].keys():
                        ft_attr[value] = { given_class: 0 }
                    else:
                        ft_attr[value][given_class] = 0

        for i in range(0, len(dfs_train)):
            example = dfs_train[i].split(',')

            if example[pos_by_attr[attr]] not in ft_attr.keys():
                ft_attr[example[pos_by_attr[attr]]] = { example[CLASS_POSITION]: 1 }
            else:
                if not ft_attr[example[pos_by_attr[attr]]].keys():
                    ft_attr[example[pos_by_attr[attr]]] = { example[CLASS_POSITION]: 1 }
                elif example[CLASS_POSITION] not in ft_attr[example[pos_by_attr[attr]]]:
                    ft_attr[example[pos_by_attr[attr]]][example[CLASS_POSITION]] = 1
                else:
                    ft_attr[example[pos_by_attr[attr]]][example[CLASS_POSITION]] = ft_attr[example[pos_by_attr[attr]]][example[CLASS_POSITION]] + 1

        total_val_sum = 0
        for key in ft_attr.keys():
            total = 0

            for given_class in ft_attr[key].keys():
                total = total + ft_attr[key][given_class]

            ft_attr[key]['total'] = total
            total_val_sum = total_val_sum + total

        ft_attr['total'] = total_val_sum
        ft[attr] = ft_attr

def build_all_frequency_tables(dfs_train, values):
    ft = {}

    for attr in attr_by_pos:
        if attr in values or attr == attr_by_pos[CLASS_POSITION]:
            build_frequency_table(dfs_train, attr, ft)

    return ft

def gain(ft, attr):
    return entropy_one_attr(ft, attr_by_pos[CLASS_POSITION]) - entropy_two_attr(ft, attr)

def best_attr_comp(ft, attrs):
    best_gain = 0
    best_attr = None

    for attr in attrs:
        if attr != attr_by_pos[CLASS_POSITION]:
            temp = gain(ft, attr)

            if temp >= best_gain:
                best_gain = temp
                best_attr = attr

    return best_attr

def if_dups(ft, parent_ft, best_gain):
    attr_to_compare = []

    if parent_ft != None:
        for attr in ft:
            if attr != attr_by_pos[CLASS_POSITION]:
                temp = gain(ft, attr)

                if temp == best_gain:
                    attr_to_compare.append(attr)

    return attr_to_compare

def get_best_attr(ft, parent_ft):
    best_gain = 0
    best_attr = None

    for attr in ft:
        if attr != attr_by_pos[CLASS_POSITION]:
            temp = gain(ft, attr)

            if temp >= best_gain:
                best_gain = temp
                best_attr = attr

    attr_to_compare = if_dups(ft, parent_ft, best_gain)

    if attr_to_compare:
        best_attr = best_attr_comp(parent_ft, attr_to_compare)

    return best_attr

def separate_dfs_by_attr(dfs_train, attr):
    if attr == None:
        return

    values = attr_values[pos_by_attr[attr]]

    dfs = {}

    for i in values:
        for example in dfs_train:
            temp = example.split(',')

            if temp[pos_by_attr[attr]] == i:
                if i not in dfs.keys():
                    dfs[i] = [ example ]
                else:
                    dfs[i].append(example)

    return dfs

def is_pure(data):
    for i in range(0, len(data) - 1):
        if data[i][CLASS_POSITION] != data[i + 1][CLASS_POSITION]:
            return False

    return True

def most_probable(ft):
    max_prob = 0
    best_attr = None

    for c in attr_values[CLASS_POSITION]:
        prob = ft[attr_by_pos[CLASS_POSITION]][c] / ft[attr_by_pos[CLASS_POSITION]]['total']

        if prob > max_prob:
            best_attr = c
            max_prob = prob

    return best_attr

def id3_clean(node, data):
    if not node:
        node = Node()
        node.values = attr_by_pos

    ft = build_all_frequency_tables(data, node.values)
    node.ft = ft

    if is_pure(data):
        example = data[0].split(',')
        node.attr = example[CLASS_POSITION]
        return node

    if len(node.values) == 1:
        node.attr = most_probable(node.ft)

        return node

    attr = get_best_attr(ft, None)
    dfs = separate_dfs_by_attr(data, attr)

    node.attr = attr
    values = node.values.copy()
    if attr != None:
        values.remove(attr)

    for val in dfs:
        child = Node()
        child.parent = node
        child.value = val
        child.values = values
        node.children.append(child)

        id3_clean(child, dfs[val])

    return node

# Pre-pruning, K = 15 - using examples count
def id3(node, data):
    if not node:
        node = Node()
        node.values = attr_by_pos

    ft = build_all_frequency_tables(data, node.values)
    node.ft = ft

    if is_pure(data):
        example = data[0].split(',')
        node.attr = example[CLASS_POSITION]
        return node

    if len(data) < 15:
        if node.parent != None:
            node.attr = most_probable(node.parent.ft)
        else:
            node.attr = most_probable(node.ft)

        return node

    if len(node.values) == 1:
        node.attr = most_probable(node.ft)

        return node

    attr = get_best_attr(ft, None)
    dfs = separate_dfs_by_attr(data, attr)

    node.attr = attr
    values = node.values.copy()
    if attr != None:
        values.remove(attr)

    for val in dfs:
        child = Node()
        child.parent = node
        child.value = val
        child.values = values
        node.children.append(child)

        id3(child, dfs[val])

    return node

# Pre-pruning, depth = 2
def id3_depth(node, data, depth):
    if not node:
        node = Node()
        node.values = attr_by_pos

    ft = build_all_frequency_tables(data, node.values)
    node.ft = ft

    if is_pure(data):
        example = data[0].split(',')
        node.attr = example[CLASS_POSITION]
        return node

    #
    if len(node.values) == 1:
        node.attr = most_probable(node.ft)

        return node

    attr = get_best_attr(ft, None)
    dfs = separate_dfs_by_attr(data, attr)

    if depth > 1:
        node.attr = most_probable(node.ft)

        return node

    node.attr = attr
    values = node.values.copy()
    if attr != None:
        values.remove(attr)

    for val in dfs:
        child = Node()
        child.parent = node
        child.value = val
        child.values = values
        node.children.append(child)

        id3_depth(child, dfs[val], depth + 1)

    return node

def test_example(tree, example):
    temp = example.split(',')

    node = tree
    while node is not None:
        if node.attr in attr_values[CLASS_POSITION]:
            res = [node.attr, temp[CLASS_POSITION]]
            return res
        else:
            check_next = node.get_child_by_value(temp[pos_by_attr[node.attr]])

            if check_next == None:
                attr = most_probable(node.ft)
                res = [attr, temp[CLASS_POSITION]]

                return res
            else:
                node = check_next

def test_model(tree, dfs_test):
    res = []

    for example in dfs_test:
        res.append(test_example(tree, example))

    return res

def k_fold_test_model_clean(dfs_test, dfs_train):
    res = []

    for index in range(0, 7):
        tree = id3_clean(None, dfs_train[index])
        # tree.display()

        res.append(test_model(tree, dfs_test[index]))

    return res

def k_fold_test_model(dfs_test, dfs_train):
    res = []

    for index in range(0, 7):
        tree = id3(None, dfs_train[index])
        # tree.display()

        res.append(test_model(tree, dfs_test[index]))

    return res

def k_fold_test_model_depth(dfs_test, dfs_train):
    res = []

    for index in range(0, 7):
        tree = id3_depth(None, dfs_train[index], 0)
        # tree.display()

        res.append(test_model(tree, dfs_test[index]))

    return res

def print_accuracy_clean(dfs_test, dfs_train):
    print('\tNo pruning')
    results = k_fold_test_model_clean(dfs_test, dfs_train)

    model_num = 0
    average = 0
    for list in results:
        correct = 0
        wrong = 0

        for res in list:
            if res[0] == res[1]:
                correct = correct + 1
            else:
                wrong = wrong + 1

        total = correct + wrong
        accuracy = correct / total
        average = average + accuracy
        model_num = model_num + 1

        print(f'Model {model_num} accuracy: {accuracy:.5f}')

    average = average / 7
    print(f'Model average accuracy: {average:.5f}')

def print_accuracy(dfs_test, dfs_train):
    print('\tPre-pruning')
    print('\t  K = 15\n')
    results = k_fold_test_model(dfs_test, dfs_train)

    model_num = 0
    average = 0
    for list in results:
        correct = 0
        wrong = 0

        for res in list:
            if res[0] == res[1]:
                correct = correct + 1
            else:
                wrong = wrong + 1

        total = correct + wrong
        accuracy = correct / total
        average = average + accuracy
        model_num = model_num + 1

        print(f'Model {model_num} accuracy: {accuracy:.5f}')

    average = average / 7
    print(f'Model average accuracy: {average:.5f}')

def print_accuracy_depth(dfs_test, dfs_train):
    print('\tPre-pruning')
    print('\t Depth = 2\n')
    results = k_fold_test_model_depth(dfs_test, dfs_train)

    model_num = 0
    average = 0
    for list in results:
        correct = 0
        wrong = 0

        for res in list:
            if res[0] == res[1]:
                correct = correct + 1
            else:
                wrong = wrong + 1

        total = correct + wrong
        accuracy = correct / total
        average = average + accuracy
        model_num = model_num + 1

        print(f'Model {model_num} accuracy: {accuracy:.5f}')

    average = average / 7
    print(f'Model average accuracy: {average:.5f}')

if __name__ == '__main__':
    with open('./data/breast-cancer.data', 'r') as data_file:
        data = data_file.read().splitlines()

    random.shuffle(data)

    # k-fold cross-validation, k = 7
    lines_count = len(data)
    size_df = math.floor(lines_count / 7)
    adjustments_cnt = lines_count % 7

    df = create_dfs(data, lines_count, size_df, adjustments_cnt)

    dfs_test = []
    dfs_train = []
    separate_dfs(df, dfs_test, dfs_train)

    print_accuracy_clean(dfs_test, dfs_train)
    print()
    print_accuracy(dfs_test, dfs_train)
    print()
    print_accuracy_depth(dfs_test, dfs_train)
