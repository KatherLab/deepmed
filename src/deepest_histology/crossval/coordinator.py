from ..experiment import Coordinator
from ..basic import train, deploy
from .get_runs import get_runs

crossval = Coordinator(get=get_runs, train=train, deploy=deploy)