import torch

class ExponentialMovingAverage:
    def __init__(self, parameters, decay: float=0.999):
        """
        parameters: параметры модели, полученные вызовом model.parameters()
        decay: параметр lambda, регулирует скорость забывания старой информации
        """
        super().__init__()

        self.decay = decay
        self.parameters = [p.clone().detach() for p in parameters if p.requires_grad]

    @torch.no_grad()
    def update(self, parameters):
        """
        Обновляет текущие параметры EMA
        parameters: обновленные параметры модели
        """

        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.parameters, parameters):
            s_param.sub_((1.0 - self.decay) * (s_param - param))

    def state_dict(self) -> dict:
        """
        Возвращает словарь с текущими параметрами EMA
        """
        return {"parameters": self.parameters}

    def copy_to(self, parameters):
        """
        Копирует сохраненные параметры в полученные параметры
        parameters: параметры модели для копирования
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.parameters, parameters):
            param.data.copy_(s_param.data)
