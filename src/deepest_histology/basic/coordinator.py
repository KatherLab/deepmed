from ..experiment import Coordinator
from .get_runs import get_runs
from .train import train
from .deploy import deploy

train_test = Coordinator(get=get_runs, train=train, deploy=deploy)
train_only = Coordinator(get=get_runs, train=train, deploy=None)
deploy_only = Coordinator(get=get_runs, train=None, deploy=deploy)