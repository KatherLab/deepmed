from ..experiment import Coordinator
from ..basic import train, deploy
from . import get_runs, evaluate

crossval = Coordinator(get=get_runs, train=train, deploy=deploy, evaluate=evaluate)