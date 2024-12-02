import onnx
from onnxsim import simplify

def transfer2sim(srcpath, despath):
    model = onnx.load(srcpath)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, despath)

if __name__ == '__main__':
    srcpath = '..\\model\\onnx\\MeModel_Kernel_T.onnx'
    filepath = ('..\\model\\onnx\\MeModel_Kernel_T_simplify.onnx')
    transfer2sim(srcpath, filepath)
    # model = onnx.load(filepath)
    # onnx.checker.check_model(model)
    # print(onnx.helper.printable_graph(model.graph))