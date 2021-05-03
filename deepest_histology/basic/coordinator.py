from ..experiment import Coordinator
from .get_runs import create_runs
from .train import train
from .deploy import deploy

basic = Coordinator(get=create_runs, train=train, deploy=deploy, evaluate=None) #TODO