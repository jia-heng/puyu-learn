import onnx
from onnxsim import simplify
import onnxslim

def transfer2sim(srcpath, despath):
    model = onnx.load(srcpath)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, despath)

def transfer2slim(srcpath, despath):
    optimized_model = onnxslim.optimize(srcpath)
    optimized_model.save(despath)

if __name__ == '__main__':
    srcpath = '..\\model\\onnx\\MeModel_Kernel_T.onnx'
    filepath = ('..\\model\\onnx\\MeModel_Kernel_T_slim.onnx')
    transfer2slim(srcpath, filepath)
    # model = onnx.load(filepath)
    # onnx.checker.check_model(model)
    # print(onnx.helper.printable_graph(model.graph))