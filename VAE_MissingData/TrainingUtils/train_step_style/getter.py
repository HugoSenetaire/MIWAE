from .trainer_step_default import TrainerStepDefault
from .grad import TrainerStepFunctorchGradNorm, TrainerStepBackpackBatchL2, TrainerStepBackpackBatch, TrainerStepManualGradNorm, TrainerStepProportionOnVal
def get_trainer_step(sampler_name, statistic_calculation, ):
    if sampler_name in ["StratifiedSamplerLipschitz", "StratifiedSamplerProportional", "StrataPersonalizedSampler",]:
        return TrainerStepDefault
    elif sampler_name in ["StratifiedSamplerAdaptiveSampling", "StratifiedSamplerAdaptiveSamplingWithFischer",]:


        # elif  statistic_calculation == "manual" :
        #     return TrainerStepManualGradNorm
        if statistic_calculation == "functorch":
            return TrainerStepFunctorchGradNorm
        elif statistic_calculation == "second_order":
            return TrainerStepBackpackBatchL2
        elif statistic_calculation == "first_order":
            return TrainerStepBackpackBatch
        elif statistic_calculation == "manual" :
            return TrainerStepManualGradNorm
        elif statistic_calculation == "validation":
            return TrainerStepProportionOnVal
        else :
            raise ValueError("Unknown statistic_calculation {}".format(statistic_calculation))
    else :
        return TrainerStepDefault