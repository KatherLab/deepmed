from ..experiment import Coordinator
from ..basic import train, deploy
from . import create_runs, evaluate

crossval = Coordinator(get=create_runs, train=train, deploy=deploy, evaluate=evaluate)