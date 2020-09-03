import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

window.onload = () => {
  const heights = [150, 160, 170]
  const weights = [40, 50, 60]

  tfvis.render.scatterplot(
    { name: '身高体重训练数据' },
    { values: heights.map((x, i) => {
      return { x, y: weights[i] }
    }) },
    {
      xAxisDomain: [140, 180],
      yAxisDomain: [30, 70]
    }
  )

  const inputs = tf.tensor(heights).sub(150).div(170-150)
  inputs.print()
  const labels = tf.tensor(weights).sub(40).div(60-40)
  labels.print()

  const model = tf.sequential()
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [1]
  }))
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.sgd(0.1)
  })

 model.fit(inputs, labels, {
    batchSize: 3,
    epochs: 100,
    callbacks: tfvis.show.fitCallbacks(
      { name: '训练过程' },
      ['loss']
    )
  }).then(() => {
    const outputs = model.predict(tf.tensor([180]).sub(150).div(20))
    alert(`身高为180cm时，预测的体重为${outputs.mul(20).add(40).dataSync()[0]}`)
  })
}