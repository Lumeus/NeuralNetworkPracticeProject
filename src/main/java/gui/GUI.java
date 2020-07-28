package gui;

import com.sun.javafx.iio.ImageFrame;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import javax.swing.*;
//import java.awt.event.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;

public class GUI extends JFrame{
	private  JButton  btnOpenFile = new JButton("choose file");
	private  JButton  btnOpenModel = new JButton("choose model");
	private  JButton  btnPredict = new JButton("predict");
	private JTextArea area = new JTextArea(1, 1);
	private JLabel label = new JLabel();

	public File net = null;
	public INDArray image = null;
	public Image image1 = null;
	public MultiLayerNetwork model = null;
	NativeImageLoader loader = new NativeImageLoader(50, 50, 3);

	public GUI (){
		super("NeuralNetworkPracticeProject");
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.setSize(500, 500);
		this.setVisible(true);
		JPanel panel = new JPanel();
		panel.setLayout(new FlowLayout());
		panel.add(btnOpenModel);
		panel.add(btnOpenFile);
		panel.add(btnPredict);
		panel.add(area);
		setContentPane(panel);


		JFileChooser fileChooser = new JFileChooser();

		btnOpenModel.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				fileChooser.setDialogTitle("Выбор модели");
				fileChooser.showOpenDialog(panel);
				net = fileChooser.getSelectedFile();
				// Если директория выбрана, покажем ее в сообщении
				net = fileChooser.getSelectedFile();
				try {
					model = ModelSerializer.restoreMultiLayerNetwork(net);
				} catch (IOException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
			}
		});

		btnOpenFile.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				fileChooser.setDialogTitle("Выбор файла");
				// Определение режима - только каталог
				int result = fileChooser.showOpenDialog(panel);
				// Если директория выбрана, покажем ее в сообщении
				if (result == JFileChooser.APPROVE_OPTION ) {
					JOptionPane.showMessageDialog(panel, fileChooser.getSelectedFile());
					File file = fileChooser.getSelectedFile();

					try {
						image1 = ImageIO.read(file);
					} catch (IOException ex) {
						ex.printStackTrace();
					}

					try {
						image = loader.asMatrix(file.getAbsolutePath());
					} catch (IOException e1) {
						e1.printStackTrace();
					}
				}

			}
		});

		btnPredict.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				String message = "";
				INDArray output = model.output(image, false);
				Integer max = output.argMax().getInt();
				
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
				//JOptionPane.showMessageDialog(null,"На картинке " + message);
				area.setText("На картинке " + message);
				
				//area.setText(output.toString());
				label.setIcon(new ImageIcon(image1.getScaledInstance(400, 300, 1)));
				
				panel.add(label);
			}
		});
	}
}
