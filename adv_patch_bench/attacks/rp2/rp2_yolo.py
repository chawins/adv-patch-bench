from adv_patch_bench.attacks.rp2 import rp2_base


class RP2AttackYOLO(rp2_base.RP2AttackModule):
    def __init__(self, attack_config, core_model, **kwargs):
        super().__init__(attack_config, core_model, **kwargs)

    def _loss_func(self, adv_img, obj_class, metadata):
        """Compute loss for YOLO models"""
        # Compute logits, loss, gradients
        out, _ = self.core_model(adv_img, val=True)
        conf = out[:, :, 4:5] * out[:, :, 5:]
        conf, labels = conf.max(-1)
        if obj_class is not None:
            loss = 0
            # Loop over EoT batch
            for c, l in zip(conf, labels):
                c_l = c[l == obj_class]
                if c_l.size(0) > 0:
                    # Select prediction from box with max confidence and ignore
                    # ones with already low confidence
                    # loss += c_l.max().clamp_min(self.min_conf)
                    loss += c_l.clamp_min(self.min_conf).sum()
            loss /= self.num_eot
        else:
            # loss = conf.max(1)[0].clamp_min(self.min_conf).mean()
            loss = conf.clamp_min(self.min_conf).sum()
            loss /= self.num_eot
        return loss
