import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { getData } from './data'

window.onload = () => {
  const data = getData(400)
  // console.log(data)

  tfvis.render.scatterplot(
    { name: '逻辑回归训练集' },
    { values: [data.filter(item => item.label === 1), data.filter(item => item.label === 0)] }
  )

  const model = tf.sequential()
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [2],
    activation: 'sigmoid'
  }))
  model.compile({
    loss: tf.losses.logLoss,
    // 优化器选择adam，会自己调节学习率
    optimizer: tf.train.adam(0.1)
  })

  const inputs = tf.tensor(data.map(item => [item.x, item.y]))
  const labels = tf.tensor(data.map(item => item.label))
  model.fit(inputs, labels, {
    batchSize: 40,
    epochs: 15,
    callbacks: tfvis.show.fitCallbacks(
      { name: '训练过程' },
      ['loss']
    )
  }).then(() => {
    isLoading = false
  })

  let isLoading = true
  const form = document.getElementById('form')
  const p = document.getElementById('result')
  form.addEventListener('submit', function (e) {
    // 取消表单默认提交行为
    e.preventDefault()
    if (!isLoading) {
      // tensor接受num类型的输入
      // 预测的tensor的shape要和训练时的shape一致
      const result = model.predict(tf.tensor([[parseInt(this.x.value), parseInt(this.y.value)]]))
      console.log(result.dataSync())
      p.innerHTML = `预测值为${result.dataSync()[0]}`
      return false
    } else {
      p.innerHTML = '模型正在加载中，请稍后...'
    }
    return false
  })
}

