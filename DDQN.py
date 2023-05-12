import numpy as np

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
from collections import deque
from keras import layers, models
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from dataget import data_test
from envofDQN import DJSPENV

class Agent:
    def __init__(self):
        self.Hid_Size = 32
        # ------------Hidden layer=5   32 nodes each layer--------------
        model = models.Sequential()
        model.add(layers.Input(shape=(4)))#352-4
        model.add(layers.Dense(self.Hid_Size, name='l1'))
        model.add(layers.Dense(self.Hid_Size, name='l2'))
        model.add(layers.Dense(self.Hid_Size, name='l3'))
        model.add(layers.Dense(self.Hid_Size, name='l4'))
        model.add(layers.Dense(self.Hid_Size, name='l5'))
        model.add(layers.Dense(6, name='l6'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=0.001))  # 0.001
        # # model.summary()
        self.model = model

        # ------------Q-network Parameters-------------
        self.act_dim = [1, 2, 3, 4, 5, 6]  # 神经网络的输出节点
        # self.obs_n = [0, 0, 0, 0, 0, 0, 0]  # 神经网路的输入节点
        self.gama = 0.95  # γ经验折损率
        # self.lr = 0.001  # 学习率
        self.global_step = 0
        self.update_target_steps = 20  # 更新目标函数的步长
        self.target_model = self.model

        # -------------------Agent-------------------
        self.e_greedy = 0.6
        self.e_greedy_decrement = 0.0001  # 0.0001
        self.L = 400  # Number of training episodes L 40

        # ---------------Replay Buffer---------------
        self.buffer = deque(maxlen=2000)
        self.Batch_size = 10  # Batch Size of Samples to perform gradient descent 40

    def replace_target(self):
        self.target_model.get_layer(name='l1').set_weights(self.model.get_layer(name='l1').get_weights())
        self.target_model.get_layer(name='l2').set_weights(self.model.get_layer(name='l2').get_weights())
        self.target_model.get_layer(name='l3').set_weights(self.model.get_layer(name='l3').get_weights())
        self.target_model.get_layer(name='l4').set_weights(self.model.get_layer(name='l4').get_weights())
        self.target_model.get_layer(name='l5').set_weights(self.model.get_layer(name='l5').get_weights())
        self.target_model.get_layer(name='l6').set_weights(self.model.get_layer(name='l6').get_weights())

    def replay(self):
        if self.global_step % self.update_target_steps == 0:
            self.replace_target()
        # replay the history and train the model
        minibatch = random.sample(self.buffer, self.Batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                k = self.target_model.predict([next_state])
                target = (reward + self.gama *
                          np.argmax(self.target_model.predict([next_state])))
            target_f = self.model.predict([state])
            # print(action,target_f,k)
            target_f[0][action] = target
            state = np.array([state])
            self.model.fit(state, target_f, epochs=1, verbose=0)
        self.global_step += 1

    def Select_action(self, obs):
        # obs=np.expand_dims(obs,0)
        if random.random() < self.e_greedy:
            act = random.randint(0, 5)
            act = np.array(act)
            # print(act,'77777',type(act))
        else:
            # print(obs)
            g = self.model.predict([obs])
            # print(g)
            act = np.argmax(g)
            # print(act,'66666')
        self.e_greedy = max(0.01, self.e_greedy - self.e_greedy_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def _append(self, exp):
        self.buffer.append(exp)

    def main(self, data):
        k = 0
        x = []
        Total_tard = []
        TR = []
        ter_y_t_time = []  # 存延期脱期
        ter_bt = []
        Sit = DJSPENV(data)
        for i in range(self.L):
            Total_reward = 0
            x.append(i + 1)
            print('-----------------------开始第', i + 1, '次训练------------------------------')
            # 把这里的Sit创建放到了外面
            obs = Sit.reset()
            obs = obs.detach().numpy().tolist()
            for il in range(Sit.op_n):
                # print(Sit.op_n)
                k += 1
                # print(obs)
                at = self.Select_action(obs)
                # print(at,'---------',type(at))
                if at == 0:
                    at_trans = Sit.rule1()
                if at == 1:
                    at_trans = Sit.rule2()
                if at == 2:
                    at_trans = Sit.rule3()
                if at == 3:
                    at_trans = Sit.rule4()
                if at == 4:
                    at_trans = Sit.rule5()
                if at == 5:
                    at_trans = Sit.rule6()
                # at_trans=self.act[at]
                print(f'这是第 {il}步>>执行action:{at}将工件序号为{at_trans}的工序加工')
                r_t, obs_t, done = Sit.scheduling(at_trans, il, Sit.op_n - 1)
                obs, at, r_t, obs_t, done = obs, at, r_t, obs_t.detach().numpy().tolist(), done
                #                 obs_t = Sit.Features()
                #                 if i == O_num - 1:
                #                     done = True
                #                 # obs = obs_t
                #                 obs_t = np.expand_dims(obs_t, 0)
                #                 # obs = np.expand_dims(obs, 0)
                #                 # print(obs,obs_t)
                #                 r_t = Sit.reward(obs[0][6], obs[0][5], obs_t[0][6], obs_t[0][5], obs[0][0], obs_t[0][0])

                self._append((obs, at, r_t, obs_t, done))
                if k > self.Batch_size:
                    # batch_obs, batch_action, batch_reward, batch_next_obs,done= self.sample()
                    self.replay()
                Total_reward += r_t
                obs = obs_t
                #                 if done == True:
                #                     break
                if il == Sit.op_n - 1:

                    ter_y_t_time.append(Sit.yttime)
                    ter_bt.append(Sit.bt)

                    break
                else:
                    continue
            #            total_tadiness = 0
            #             Job = Sit.n
            #             E = 0
            #             K = [i for i in range(len(Job))]
            #             End = []
            #             for Ji in range(len(Job)):
            #                 End.append(max(Job[Ji].End))
            #                 if max(Job[Ji].End) > D[Ji]:

            #                     total_tadiness += abs(max(Job[Ji].End) - D[Ji])
            #             print('<<<<<<<<<-----------------total_tardiness:', total_tadiness, '------------------->>>>>>>>>>')
            #             Total_tard.append(total_tadiness)
            print('<<<<<<<<<-----------------reward:', Total_reward, '------------------->>>>>>>>>>')
            TR.append(Total_reward)
        #             # plt.plot(K,End,color='y')
        #             # plt.plot(K,D,color='r')
        #             # plt.show()
        with open("tt.txt",'a') as f:
            for i in TR:
                f.write(str(i)+',')
            f.write('\n')
            for i in ter_y_t_time:
                f.write(str(i)+',')
            f.write('\n')
            for i in ter_bt:
                f.write(str(i)+',')
            f.write('\n')
        # plt.figure(1)
        # plt.plot(x, TR)
        # plt.savefig('re14')
        #
        # plt.show()
        # plt.figure(2)
        # plt.plot(x, ter_y_t_time)
        # plt.savefig('yttime14')
        # plt.show()
        #
        # plt.figure(3)
        # plt.plot(x, ter_bt)
        # plt.savefig('diedaishoulian14')
        # plt.show()

        return Total_reward

# for i in range(8):
#     print(i)
#     st = Agent()
#     st.main(data_test)
# 不带弧线，且奖励函数就是当前利用率！rou=0.6
st = Agent()
st.main(data_test)
# with open('minfinish.txt','a') as fp:
#     fp.write('\n')