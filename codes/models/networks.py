from .modules.sttrans import STTrans2
import models.modules.sttrans as sttrans

####################
# define network
####################
# Generator


def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'STTrans2':
        netG = sttrans.STTrans2(n_outputs=opt_net['nframes'], nf=opt_net['nf'], embed_dim=opt_net['embed_dim'])
        # netG = STTrans2(n_outputs=opt_net['nframes'], nf=opt_net['nf'], embed_dim=opt_net['embed_dim'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
