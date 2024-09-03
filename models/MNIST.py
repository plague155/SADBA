import torch
import torch.nn as nn
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self,name,created_time):
        super(SimpleNet, self).__init__()
        self.conv1=nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.fc1=nn.Linear(64*14*14,128)
        self.fc2=nn.Linear(128,10)
        self.relu=nn.ReLU()
        self.name=name
        self.created_time=created_time

    def forward(self,x):
        x=self.relu(self.conv1(x))
        x=self.pool(self.conv2(x))
        x=x.view(-1,64*14*14)
        x=self.relu(self.fc1(x))
        x=self.fc2(x)
        return x

    def track_distance_batch_vis(self, vis, epoch, data_len, batch, distance_to_global_model, eid, name=None,
                                 is_poisoned=False):
        x = (epoch - 1) * data_len + batch + 1

        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name
        else:
            name = name + '_poisoned' if is_poisoned else name

        vis.line(Y=np.array([distance_to_global_model]), X=np.array([x]),
                 win=f"global_dist_{self.created_time}",
                 env=eid,
                 name=f'Model_{name}',
                 update='append' if
                 vis.win_exists(f"global_dist_{self.created_time}",
                                env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Distance to Global {self.created_time}",
                           width=700, height=400))

    def trigger_agent_test_vis(self, vis, epoch, acc, loss, eid, name):
        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"poison_state_trigger_acc_{self.created_time}",
                 env=eid,
                 name=f'{name}',
                 update='append' if vis.win_exists(f"poison_state_trigger_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Backdoor State Trigger Test Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"poison_state_trigger_loss_{self.created_time}",
                     env=eid,
                     name=f'{name}',
                     update='append' if vis.win_exists(f"poison_state_trigger_loss_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Backdoor State Trigger Test Loss_{self.created_time}",
                               width=700, height=400))

    def train_vis(self, vis, epoch, acc, loss=None, eid='main', is_poisoned=False, name=None):
        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name
        vis.line(X=np.array([epoch]), Y=np.array([acc]), name=name, win='train_acc_{0}'.format(self.created_time),
                 env=eid,
                 update='append' if vis.win_exists('train_acc_{0}'.format(self.created_time), env=eid) else None,
                 opts=dict(showlegend=True, title='Train Accuracy_{0}'.format(self.created_time),
                           width=700, height=400))
        if loss is not None:
            vis.line(X=np.array([epoch]), Y=np.array([loss]), name=name, env=eid,
                     win='train_loss_{0}'.format(self.created_time),
                     update='append' if vis.win_exists('train_loss_{0}'.format(self.created_time), env=eid) else None,
                     opts=dict(showlegend=True, title='Train Loss_{0}'.format(self.created_time), width=700,
                               height=400))
        return

    def test_vis(self, vis, epoch, acc, loss, eid, agent_name_key):
        name = agent_name_key
        # name= f'Model_{name}'

        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"test_acc_{self.created_time}",
                 env=eid,
                 name=name,
                 update='append' if vis.win_exists(f"test_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Main Task Test Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"test_loss_{self.created_time}",
                     env=eid,
                     name=name,
                     update='append' if vis.win_exists(f"test_loss_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Main Task Test Loss_{self.created_time}",
                               width=700, height=400))

    def poison_test_vis(self, vis, epoch, acc, loss, eid, agent_name_key):
        name = agent_name_key
        # name= f'Model_{name}'

        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"poison_test_acc_{self.created_time}",
                 env=eid,
                 name=name,
                 update='append' if vis.win_exists(f"poison_test_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Backdoor Task Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"poison_loss_acc_{self.created_time}",
                     env=eid,
                     name=name,
                     update='append' if vis.win_exists(f"poison_loss_acc_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Backdoor Task Test Loss_{self.created_time}",
                               width=700, height=400))