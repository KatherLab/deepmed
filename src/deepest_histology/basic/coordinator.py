from ..experiment import Coordinator
from .get_runs import get_runs
from .train import train
from .deploy import deploy
from .evaluate import evaluate

train_test = Coordinator(get=get_runs, train=train, deploy=deploy, evaluate=evaluate)
train_only = Coordinator(get=get_runs, train=train, deploy=None, evaluate=None)
deploy_only = Coordinator(get=get_runs, train=None, deploy=deploy, evaluate=evaluate)