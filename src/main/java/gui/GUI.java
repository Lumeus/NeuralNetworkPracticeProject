// Класс, реализующий графический интерфейс

package gui;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;

public class GUI extends JFrame{
	private  JButton  btnOpenFile = new JButton("choose file"); // Кнопка выбора изображения
	private  JButton  btnOpenModel = new JButton("choose model"); // Кнопка выбора модели
	private  JButton  btnPredict = new JButton("predict"); // Кнопка запуска классификации
	private JTextArea area = new JTextArea(1, 1); // Текстовое поле
	private JLabel label = new JLabel(); // Поле вывода изображения

	public File net = null; // Файл конфигурации модели
	public INDArray image = null; // Массив данных для модели
	public Image image1 = null; // Изображение для вывода
	public MultiLayerNetwork model = null; // Модель
	NativeImageLoader loader = new NativeImageLoader(50, 50, 3); // Загрузчик

	// Конструктор
	public GUI (){
		super("NeuralNetworkPracticeProject"); // Заголовок окна
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // Назначение кнопки выхода
		this.setSize(500, 500); // Размер окна
		this.setVisible(true);
		// Добавление необходимых элементов в окно
		JPanel panel = new JPanel();
		panel.setLayout(new FlowLayout());
		panel.add(btnOpenModel);
		panel.add(btnOpenFile);
		panel.add(btnPredict);
		panel.add(area);
		setContentPane(panel);

		JFileChooser fileChooser = new JFileChooser();

		// Кнопка выбора модели
		btnOpenModel.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				// Окно выбора модели
				fileChooser.setDialogTitle("Выбор модели");
				fileChooser.showOpenDialog(panel);
				int result = fileChooser.showOpenDialog(panel);
				// Если директория выбрана, покажем ее в сообщении
				if (result == JFileChooser.APPROVE_OPTION ) {
					JOptionPane.showMessageDialog(panel, fileChooser.getSelectedFile());
					// Открытие файла
					net = fileChooser.getSelectedFile();
					// Чтение модели
					try {
						model = ModelSerializer.restoreMultiLayerNetwork(net);
					} catch (IOException e1) {
						e1.printStackTrace();
					}
				}
			}
		});

		// Кнопка выбора изображения
		btnOpenFile.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				fileChooser.setDialogTitle("Выбор файла");
				int result = fileChooser.showOpenDialog(panel);
				// Если директория выбрана, покажем ее в сообщении
				if (result == JFileChooser.APPROVE_OPTION ) {
					JOptionPane.showMessageDialog(panel, fileChooser.getSelectedFile());
					// Откратие файла
					File file = fileChooser.getSelectedFile();
					// Чтение изображения
					try {
						image1 = ImageIO.read(file);
					} catch (IOException ex) {
						ex.printStackTrace();
					}
					// Чтение данных для модели
					try {
						image = loader.asMatrix(file.getAbsolutePath());
					} catch (IOException e1) {
						e1.printStackTrace();
					}
				}

			}
		});

		// Кнопка запуска классификации
		btnPredict.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				String message = "";
				// Получение предсказания
				INDArray output = model.output(image, false);
				Integer max = output.argMax().getInt();
				// Интерпретация предсказания
				switch (max){
					case(0):
						message = "ГРЕЧКА";
						break;
					case(1):
						message = "ОВСЯНКА";
						break;
					case(2):
						message = "ПЕРЛОВКА";
						break;
					case(3):
						message = "РИС";
						break;
				}
				// Вывод предсказания
				area.setText("На картинке " + message);
				// Вывод изображения
				label.setIcon(new ImageIcon(image1.getScaledInstance(400, 300, 1)));
				panel.add(label);
			}
		});
	}
}
