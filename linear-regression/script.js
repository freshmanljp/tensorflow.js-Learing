import * as tf from '@tensorflow/tfjs'
import * as tfvs from '@tensorflow/tfjs-vis'

window.onload = () => {
  // 输入和标签
  const x = [1, 2, 3, 4]
  const y = [1, 3, 5, 7]
   
  // 调用tfvs的绘制散点图api
  tfvs.render.scatterplot(
    // 渲染挂载节点设置
    { name: '线性回归训练集' },
    // 散点图data设置
    { values: x.map((item, index) => ({x: item, y: y[index]})) },
    // 散点图相关配置
    { xAxisDomain: [0, 5], yAxisDomain: [0, 8] }
  )

  // 创建一个连续模型
  const model = tf.sequential()
  // 给模型添加一个全连接层
  model.add(tf.layers.dense({
    // 神经元数
    units: 1,
    // 输入的形状
    inputShape: [1]
  }))
  // 模型的训练配置
  model.compile({
    // 设置损失函数为均方误差函数
    loss: tf.losses.meanSquaredError,
    // 设置优化器为随机梯度下降，学习率为0.1
    optimizer: tf.train.sgd(0.1)
  })

  const input = tf.tensor(x)
  const label = tf.tensor(y)
  model.fit(input, label, {
    // 批处理个数，一次迭代处理多少个训练数据
    batchSize: 4,
    // 迭代次数
    epochs: 100,
    // 训练过程可视化配置
    callbacks: tfvs.show.fitCallbacks(
      { name: '训练过程' },
      // 这个配置项待理解
      ['loss']
    )
  })
  // fit方法是个异步执行api
  .then(() => {
    // 调用dataSync获取正确的数据类型
    const output = model.predict(tf.tensor([5])).dataSync()
    console.log(output)
  })
}