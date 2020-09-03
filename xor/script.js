import { getData } from './data.js'
import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { callbacks } from '@tensorflow/tfjs';

window.onload = () => {
  const data = getData(400)
  let isLoading = true
  tfvis.render.scatterplot(
    { name: 'xor训练数据集' },
    { values: [data.filter(item => item.label === 1), data.filter(item => item.label === 0)] }
  )

  const model = tf.sequential()
  // 含有四个神经元的隐藏层，激活函数为relu
  model.add(tf.layers.dense({
    units: 4,
    inputShape: [2],
    activation: 'relu'
  }))
  // 输出层，只概率一个结果，所以只需要一个神经元，输出值在0~1之间，因此用sigmoid激活函数
  model.add(tf.layers.dense({
    units: 1,
    // inputShape自动根据前一层的神经元个数生成
    activation: 'sigmoid'
  }))

  model.compile({
    loss: tf.losses.logLoss,
    optimizer: tf.train.adam(0.1)
  })

  const inputs = tf.tensor(data.map(item => [item.x, item.y]))
  const labels = tf.tensor(data.map(item => item.label))
  model.fit(inputs, labels, {
    batchSize: 40,
    epochs: 10,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'xor训练效果' },
      ['loss']
    )
  }).then(() => {
    isLoading = false
  })

  const form = document.getElementById('form')
  const p = document.getElementById('result')
  form.addEventListener('submit', function (e) {
    // 取消表单默认提交行为
    e.preventDefault()
    if (!isLoading) {
      // tensor接受num类型的输入
      // 预测的tensor的shape要和训练时的shape一致
      const result = model.predict(tf.tensor([[parseInt(this.x.value), parseInt(this.y.value)]]))
      p.innerHTML = `预测值为${result.dataSync()[0]}`
      return false
    } else {
      p.innerHTML = '模型正在加载中，请稍后...'
    }
  })
}