def run(img,template,gt):
    print(img,template,gt)

if __name__ == "__main__":
    dic = {"img":"imgdata","template":"templatedata","gt":"gtdata"}
    run(**dic)
