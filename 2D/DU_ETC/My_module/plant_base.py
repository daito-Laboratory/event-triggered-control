
class PlantBase:
    def x_dot(self, x, u):
        raise NotImplementedError("x_dotをサブクラスで定義してください")
