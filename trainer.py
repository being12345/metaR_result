import numpy as np

from models import *
from tensorboardX import SummaryWriter
import os
import sys
import torch
import shutil
import logging


def test_task(train_task):
    support, support_negative, query, negative = train_task
    print(
        f'relation_num: {len(support)} support_num: {len(support[0])} support_neg_num: {len(support_negative[0])} '
        f'query_num: {len(query[0])} query_neg_num: {len(negative[0])}')


class Trainer:
    def __init__(self, data_loaders, dataset, parameter):
        self.parameter = parameter
        # data loader
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        self.fw_dev_data_loader = data_loaders[3]
        # parameters
        self.few = parameter['few']
        self.num_query = parameter['num_query']
        self.batch_size = parameter['batch_size']
        self.learning_rate = parameter['learning_rate']
        self.early_stopping_patience = parameter['early_stopping_patience']
        # epoch
        self.epoch = parameter['epoch']
        self.print_epoch = parameter['print_epoch']
        self.eval_epoch = parameter['eval_epoch']
        self.checkpoint_epoch = parameter['checkpoint_epoch']
        # device
        self.device = parameter['device']
        self.metaR = MetaR(dataset, parameter)
        self.metaR.to(self.device)
        # optimizer
        self.optimizer = torch.optim.Adam(self.metaR.parameters(), self.learning_rate)
        # tensorboard log writer
        if parameter['step'] == 'train':
            self.writer = SummaryWriter(os.path.join(parameter['log_dir'], parameter['prefix']))
        # dir
        self.state_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'])
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        self.ckpt_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'], 'checkpoint')
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.state_dict_file = ''

        # logging
        logging_dir = os.path.join(self.parameter['log_dir'], self.parameter['prefix'], 'res.log')
        logging.basicConfig(filename=logging_dir, level=logging.INFO, format="%(asctime)s - %(message)s")
        self.csv_dir = os.path.join(self.parameter['log_dir'], self.parameter['prefix'])

        # load state_dict and params
        if parameter['step'] in ['test', 'dev']:
            self.reload()

    def reload(self):
        if self.parameter['eval_ckpt'] is not None:
            state_dict_file = os.path.join(self.ckpt_dir, 'state_dict_' + self.parameter['eval_ckpt'] + '.ckpt')
        else:
            state_dict_file = os.path.join(self.state_dir, 'state_dict')
        self.state_dict_file = state_dict_file
        logging.info('Reload state_dict from {}'.format(state_dict_file))
        print('reload state_dict from {}'.format(state_dict_file))
        state = torch.load(state_dict_file, map_location=self.device)
        if os.path.isfile(state_dict_file):
            self.metaR.load_state_dict(state)
        else:
            raise RuntimeError('No state dict in {}!'.format(state_dict_file))

    def save_checkpoint(self, epoch):
        torch.save(self.metaR.state_dict(), os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt'))

    def del_checkpoint(self, epoch):
        path = os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt')
        if os.path.exists(path):
            os.remove(path)
        else:
            raise RuntimeError('No such checkpoint to delete: {}'.format(path))

    def save_best_state_dict(self, best_epoch):
        shutil.copy(os.path.join(self.ckpt_dir, 'state_dict_' + str(best_epoch) + '.ckpt'),
                    os.path.join(self.state_dir, 'state_dict'))

    def write_training_log(self, data, task, epoch):
        self.writer.add_scalar(f'Training_Loss_{task}', data['Loss'], epoch)

    def write_fw_validating_log(self, data, record, task, epoch):
        self.writer.add_scalar(f'Few_Shot_Validating_MRR_{task}', data['MRR'], epoch)
        self.writer.add_scalar(f'Few_Shot_Validating_Hits_10_{task}', data['Hits@10'], epoch)
        self.writer.add_scalar(f'Few_Shot_Validating_Hits_5_{task}', data['Hits@5'], epoch)
        self.writer.add_scalar(f'Few_Shot_Validating_Hits_1_{task}', data['Hits@1'], epoch)

        if epoch + self.eval_epoch >= self.epoch:
            record[0][task, task] = data['MRR']
            record[1][task, task] = data['Hits@10']
            record[2][task, task] = data['Hits@5']
            record[3][task, task] = data['Hits@1']

    def write_cl_validating_log(self, metrics, record, task):
        for i, data in enumerate(metrics):
            self.writer.add_scalar(f'Continual_Learning_Validating_MRR_{task}', data['MRR'], i)
            self.writer.add_scalar(f'Continual_Learning_Validating_Hits_10_{task}', data['Hits@10'], i)
            self.writer.add_scalar(f'Continual_Learning_Validating_Hits_5_{task}', data['Hits@5'], i)
            self.writer.add_scalar(f'Continual_Learning_Validating_Hits_1_{task}', data['Hits@1'], i)
            record[0][task, i] = data['MRR']
            record[1][task, i] = data['Hits@10']
            record[2][task, i] = data['Hits@5']
            record[3][task, i] = data['Hits@1']

    def logging_fw_training_data(self, data, epoch, task):
        if epoch == self.eval_epoch:
            logging.info(f"Few_Shot_Learning_task {task}")
        logging.info("Epoch: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            epoch, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def logging_cl_training_data(self, metrics, task):
        logging.info(f"Eval_Continual_Learning_task {task}")
        for i, data in enumerate(metrics):
            logging.info("Task: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                str(i), data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def logging_eval_data(self, data, state_path, istest=False):
        setname = 'dev set'
        if istest:
            setname = 'test set'
        logging.info("Eval {} on {}".format(state_path, setname))
        logging.info("MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def rank_predict(self, data, x, ranks):
        # query_idx is the idx of positive score
        query_idx = x.shape[0] - 1
        # sort all scores with descending, because more plausible triple has higher score
        _, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        ranks.append(rank)
        # update data
        if rank <= 10:
            data['Hits@10'] += 1
        if rank <= 5:
            data['Hits@5'] += 1
        if rank == 1:
            data['Hits@1'] += 1
        data['MRR'] += 1.0 / rank

    def do_one_step(self, task, iseval=False, curr_rel=''):
        loss, p_score, n_score = 0, 0, 0
        if not iseval:
            self.optimizer.zero_grad()
            p_score, n_score = self.metaR(task, iseval, curr_rel)
            y = torch.ones(p_score.shape[0], 1).to(self.device)
            # y = torch.Tensor([1]).to(self.device)
            loss = self.metaR.loss_func(p_score, n_score, y)
            loss.backward()
            self.optimizer.step()
        elif curr_rel != '':
            p_score, n_score = self.metaR(task, iseval, curr_rel)
            y = torch.ones(p_score.shape[0], 1).to(self.device)
            # y = torch.Tensor([1]).to(self.device)
            loss = self.metaR.loss_func(p_score, n_score, y)
        return loss, p_score, n_score

    def train(self):
        # initialization
        best_epoch = 0
        best_value = 0
        bad_counts = 0
        num_tasks = 8  # TODO: update it in parser
        MRR_val_mat = np.zeros((num_tasks, num_tasks))  # record fw and cl vl MRR metrics
        Hit1_val_mat = np.zeros((num_tasks, num_tasks))  # record fw and cl vl MRR metrics
        Hit5_val_mat = np.zeros((num_tasks, num_tasks))  # record fw and cl vl MRR metrics
        Hit10_val_mat = np.zeros((num_tasks, num_tasks))  # record fw and cl vl MRR metrics
        val_mat = [MRR_val_mat, Hit1_val_mat, Hit5_val_mat, Hit10_val_mat]

        for task in range(num_tasks):
            # training by epoch
            for e in range(self.epoch):
                is_last = False if e != self.epoch - 1 else True
                is_base = True if task == 0 else False
                # sample one batch from data_loader
                train_task, curr_rel = self.train_data_loader.next_batch(is_last, is_base)
                # Test train_task num
                loss, _, _ = self.do_one_step(train_task, iseval=False, curr_rel=curr_rel)
                # if e == 0:  # TODO: test module move later
                #     test_task(train_task)

                # print the loss on specific epoch
                if e % self.print_epoch == 0:
                    loss_num = loss.item()
                    self.write_training_log({'Loss': loss_num}, task, e)
                    print("Epoch: {}\tLoss: {:.4f}".format(e, loss_num))
                # # save checkpoint on specific epoch
                # if e % self.checkpoint_epoch == 0 and e != 0:
                #     print('Epoch  {} has finished, saving...'.format(e))
                #     self.save_checkpoint(e)

                # do evaluation on specific epoch
                if e % self.eval_epoch == 0 and e != 0:
                    print('Epoch  {} has finished, validating few shot...'.format(e))
                    valid_data = self.fw_eval(task, istest=False, epoch=e)  # few shot val
                    self.write_fw_validating_log(valid_data, val_mat, task, e)
                if task != 0 and e == self.epoch - 1:
                    print('Epoch  {} has finished, validating continual learning...'.format(e))

                    valid_data = self.novel_continual_eval(previous_relation, task,
                                                           istest=False)  # continual learning val only in last epoch
                    self.write_cl_validating_log(valid_data, val_mat, task)

                    # metric = self.parameter['metric']
                    # # early stopping checking
                    # if valid_data[metric] > best_value:
                    #     best_value = valid_data[metric]
                    #     best_epoch = e
                    #     print('\tBest model | {0} of valid set is {1:.3f}'.format(metric, best_value))
                    #     bad_counts = 0
                    #     # save current best
                    #     self.save_checkpoint(best_epoch)
                    # else:
                    #     print('\tBest {0} of valid set is {1:.3f} at {2} | bad count is {3}'.format(
                    #         metric, best_value, best_epoch, bad_counts))
                    #     bad_counts += 1
                    #
                    # if bad_counts >= self.early_stopping_patience:
                    #     print('\tEarly stopping at epoch %d' % e)
                    #     break

            previous_relation = curr_rel  # cache previous relations

        np.savetxt(os.path.join(self.csv_dir, 'MRR.csv'), MRR_val_mat, delimiter=",")
        np.savetxt(os.path.join(self.csv_dir, 'Hit@10.csv'), Hit10_val_mat, delimiter=",")
        np.savetxt(os.path.join(self.csv_dir, 'Hit@5.csv'), Hit5_val_mat, delimiter=",")
        np.savetxt(os.path.join(self.csv_dir, 'Hit@1.csv'), Hit1_val_mat, delimiter=",")
        print('Training has finished')
        # print('\tBest epoch is {0} | {1} of valid set is {2:.3f}'.format(best_epoch, metric, best_value))
        # self.save_best_state_dict(best_epoch)
        print('Finish')

    def novel_continual_eval(self, previous_rel, task, istest=False):
        self.metaR.eval()
        # clear sharing rel_q
        self.metaR.rel_q_sharing = dict()

        data_loader = self.test_data_loader if istest is True else self.dev_data_loader
        # if epoch == self.eval_epoch:  # only eval cl in last epoch
        current_eval_num = 0
        for rel in previous_rel:
            data_loader.eval_triples.extend(data_loader.tasks[rel][self.few:])
            current_eval_num += len(data_loader.tasks[rel][self.few:])
        data_loader.num_tris = len(data_loader.eval_triples)
        data_loader.tasks_relations_num.append(current_eval_num)
        data_loader.curr_tri_idx = 0

        # initial return data of validation
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        tasks_data = []
        ranks = []

        t = 0
        i = 0
        previous_t = 0
        temp = dict()
        while True:
            # sample all the eval tasks
            eval_task, curr_rel = data_loader.next_one_on_eval()
            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == 'EOT':
                break
            t += 1

            self.get_epoch_score(curr_rel, data, eval_task, ranks, t, temp)

            if t == data_loader.tasks_relations_num[i] + previous_t:
                # cache task score
                tasks_data.append(data)
                # clear data
                data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
                i += 1
                previous_t = t

        # print overall evaluation result and return it
        for data in tasks_data:
            for k in data.keys():
                data[k] = round(data[k] / t, 3)

        # if self.parameter['step'] == 'train':
        #     self.logging_training_data(data, epoch)
        # else:
        #     self.logging_eval_data(data, self.state_dict_file, istest)

        print('continual learning', tasks_data)
        if self.parameter['step'] == 'train':
            self.logging_cl_training_data(tasks_data, task)
        return tasks_data

    def fw_eval(self, task, istest=False, epoch=None):
        self.metaR.eval()
        # clear sharing rel_q
        self.metaR.rel_q_sharing = dict()

        data_loader = self.test_data_loader if istest is True else self.fw_dev_data_loader
        data_loader.curr_tri_idx = 0

        # initial return data of validation
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        ranks = []

        t = 0
        temp = dict()
        while True:
            # sample all the eval tasks
            eval_task, curr_rel = data_loader.next_one_on_eval()
            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == 'EOT':
                break
            t += 1
            self.get_epoch_score(curr_rel, data, eval_task, ranks, t, temp)

        # print overall evaluation result and return it
        for k in data.keys():
            data[k] = round(data[k] / t, 3)

        if self.parameter['step'] == 'train':
            self.logging_fw_training_data(data, epoch, task)
        else:
            self.logging_eval_data(data, self.state_dict_file, istest)

        print("few shot {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            t, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

        return data

    def get_epoch_score(self, curr_rel, data, eval_task, ranks, t, temp):
        _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=curr_rel)
        x = torch.cat([n_score, p_score], 1).squeeze()
        self.rank_predict(data, x, ranks)

        # print current temp data dynamically
        for k in data.keys():
            temp[k] = data[k] / t
        sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
        sys.stdout.flush()

    def eval_by_relation(self, istest=False, epoch=None):
        self.metaR.eval()
        self.metaR.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        all_data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        all_t = 0
        all_ranks = []

        for rel in data_loader.all_rels:
            print("rel: {}, num_cands: {}, num_tasks:{}".format(
                rel, len(data_loader.rel2candidates[rel]), len(data_loader.tasks[rel][self.few:])))
            data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
            temp = dict()
            t = 0
            ranks = []
            while True:
                eval_task, curr_rel = data_loader.next_one_on_eval_by_relation(rel)
                if eval_task == 'EOT':
                    break
                t += 1

                _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=rel)
                x = torch.cat([n_score, p_score], 1).squeeze()

                self.rank_predict(data, x, ranks)

                for k in data.keys():
                    temp[k] = data[k] / t
                sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                    t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
                sys.stdout.flush()

            print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))

            for k in data.keys():
                all_data[k] += data[k]
            all_t += t
            all_ranks.extend(ranks)

        print('Overall')
        for k in all_data.keys():
            all_data[k] = round(all_data[k] / all_t, 3)
        print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            all_t, all_data['MRR'], all_data['Hits@10'], all_data['Hits@5'], all_data['Hits@1']))

        return all_data
