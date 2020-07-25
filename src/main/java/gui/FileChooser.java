package gui;

import javax.swing.*;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.event.*;
import java.io.File;
import java.io.IOException;

public class FileChooser extends JFrame{

    private  JButton  btnOpenModel = null;
    private  JButton  btnOpenFile = null;
    private  JButton  btnPredict = null;
    private JLabel label = null;
    public File net = null;
    public INDArray image = null;
    public MultiLayerNetwork model = null;
    NativeImageLoader loader = new NativeImageLoader(50, 50, 3);

    private  JFileChooser fileChooser = null;

    private final String[][] FILTERS = {{"docx", "Файлы Word (*.docx)"},
                                        {"pdf" , "Adobe Reader(*.pdf)"}};
    public FileChooser() {
        super("Пример FileChooser");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        // Кнопка создания диалогового окна для выбора директории
        btnOpenModel = new JButton("Загрузить модель");
        // Кнопка создания диалогового окна для сохранения файла
        btnOpenFile = new JButton("Открыть изображение");
        // Кнопка создания диалогового окна для сохранения файла
        btnPredict = new JButton("Предсказать");
        
        
        
        // Создание экземпляра JFileChooser 
        fileChooser = new JFileChooser();
        // Подключение слушателей к кнопкам
        //addFileChooserListeners();
        btnOpenModel.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                fileChooser.setDialogTitle("Выбор модели");
                // Определение режима - только каталог
                //fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
                int result = fileChooser.showOpenDialog(FileChooser.this);
                // Если директория выбрана, покажем ее в сообщении
                if (result == JFileChooser.APPROVE_OPTION ) {
                	JOptionPane.showMessageDialog(FileChooser.this, fileChooser.getSelectedFile());
                    net = fileChooser.getSelectedFile();
                    try {
						model = ModelSerializer.restoreMultiLayerNetwork(net);
					} catch (IOException e1) {
						// TODO Auto-generated catch block
						e1.printStackTrace();
					}
                }
            }
        });
        
        btnOpenFile.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                fileChooser.setDialogTitle("Выбор модели");
                // Определение режима - только каталог
                //fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
                int result = fileChooser.showOpenDialog(FileChooser.this);
                // Если директория выбрана, покажем ее в сообщении
                if (result == JFileChooser.APPROVE_OPTION ) {
                	JOptionPane.showMessageDialog(FileChooser.this, fileChooser.getSelectedFile());
                    File file = fileChooser.getSelectedFile();
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
        		INDArray output = model.output(image, false);
        		Integer max = output.argMax().getInt();
        		label = new JLabel(max.toString());
        	}
        });

        // Размещение кнопок в интерфейсе
        JPanel contents = new JPanel();
        contents.add(btnOpenModel);
        contents.add(btnOpenFile);
        contents.add(btnPredict);
        //contents.add(label);
        setContentPane(contents);
        // Вывод окна на экран
        setSize(360, 110);
        setVisible(true);
    }
    
    public static void main(String[] args) {
        // Локализация компонентов окна JFileChooser
        UIManager.put(
                 "FileChooser.saveButtonText", "Сохранить");
        UIManager.put(
                 "FileChooser.cancelButtonText", "Отмена");
        UIManager.put(
                 "FileChooser.fileNameLabelText", "Наименование файла");
        UIManager.put(
                 "FileChooser.filesOfTypeLabelText", "Типы файлов");
        UIManager.put(
                 "FileChooser.lookInLabelText", "Директория");
        UIManager.put(
                 "FileChooser.saveInLabelText", "Сохранить в директории");
        UIManager.put(
                 "FileChooser.folderNameLabelText", "Путь директории");

        new FileChooser();
    }
}
