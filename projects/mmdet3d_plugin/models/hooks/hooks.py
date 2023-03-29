from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time


@HOOKS.register_module()
class GradChecker(Hook):

    def after_train_iter(self, runner):
        for key, val in runner.model.named_parameters():
            if val.grad == None and val.requires_grad:
                print('WARNNING: {key}\'s parameters are not be used!!!!'.format(key=key))


