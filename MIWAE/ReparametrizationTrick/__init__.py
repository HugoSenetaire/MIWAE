from .abstract_reparam import ReparamTrick
from .reparam_normal import ReparamTrickNormal
from .reparam_bernoulli import ReparamTrickBernoulli

dic_reparametrization = {
    "ReparamTrickBernoulli" : ReparamTrickBernoulli,
    "ReparamTrickNormal" : ReparamTrickNormal,
}