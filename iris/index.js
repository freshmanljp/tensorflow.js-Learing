import { getIrisData, IRIS_CLASSES, IRIS_NUM_CLASSES } from './data.js'

window.onload = () => {
  const [x_train, y_train, x_test, y_test] = getIrisData(0.2)
  console.log(x_train)
  console.log(y_train)
  console.log(x_test)
  console.log(y_test)
}