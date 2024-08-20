import tkinter as tk
from tkinter import filedialog, messagebox
from codes import functions


class KMeansApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KMeans Clustering")

        self.file_path = None
        self.variable_names = None
        self.check_vars = None

        # Frame de seleção de arquívo
        self.FrameSelArquivo = tk.Frame(root)
        self.FrameSelArquivo.grid(row=0, column=0,pady=10)

        self.label = tk.Label(self.FrameSelArquivo, text="Nenhum arquivo selecionado")
        self.label.grid(row=0, column=0, pady=10)

        self.select_button = tk.Button(self.FrameSelArquivo, text="Selecionar Arquivo", command=self.select_file)
        self.select_button.grid(row=1, column=0, pady=10)

        # Frame edição do arquivo
        self.FrameEditArquivo = tk.Frame(root)
        self.label_depth = tk.Label(self.FrameEditArquivo, text="Digite uma profundidade:")
        self.label_depth.grid(row=0, column=0, pady=10, padx=5)
        self.entry_depth = tk.Entry(self.FrameEditArquivo)
        self.entry_depth.grid(row=0, column=1, pady=10, padx=5)

        # Frame de parâmetros do cluster
        self.FrameParamCluster = tk.Frame(root)
        self.FrameParamCluster.grid(row=1, column=0, pady=10)

        self.label_clusters = tk.Label(self.FrameParamCluster, text="Digite o número de clusters:")
        self.label_clusters.grid(row=0, column=0, pady=10, padx=5)

        self.entry_clusters = tk.Entry(self.FrameParamCluster)
        self.entry_clusters.grid(row=0, column=1, pady=10, padx=5)

        # Frame imagem
        self.FrameMapa = tk.Frame(root)
        self.FrameMapa.grid(row=2, column=0, pady=10)

        self.label_title = tk.Label(self.FrameMapa, text="Título:")
        self.label_title.grid(row=0,column=0, pady=10, padx=5)
        self.entry_title = tk.Entry(self.FrameMapa)
        self.entry_title.grid(row=0, column=1, pady=10, padx=5)

        self.var_save = tk.BooleanVar()
        self.check_save = tk.Checkbutton(self.FrameMapa, text="Salvar", variable=self.var_save)
        self.check_save.grid(row=1, column=0, pady=10, padx=10)

        # Rodar código
        self.kmeans_button = tk.Button(root, text="KMeans", command=self.apply_kmeans)
        self.kmeans_button.grid(row=3, column=0, columnspan=2, pady=10)

    def constroi_frame_edicao(self):
        self.FrameEditArquivo.grid(row=1, column=0, pady=10)
        self.FrameParamCluster.grid(row=2, column=0, pady=10)
        self.FrameMapa.grid(row=3, column=0, pady=10)
        self.kmeans_button.grid(row=4, column=0, columnspan=2, pady=10)


    def show_check_buttons(self):
        self.check_vars = {}
        check_buttons = []
        indice = 0

        for col in self.variable_names:
            var = tk.BooleanVar(value=True)
            check = tk.Checkbutton(self.FrameEditArquivo, text=col, variable=var)
            check.grid(row=1, column=indice)
            indice += 1
            self.check_vars[col] = var
            check_buttons.append(check)


    def select_file(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            self.label.config(text=self.file_path)
            self.variable_names = functions.lista_variaveis(self.file_path)
            self.show_check_buttons()
            self.constroi_frame_edicao()

    def seleciona_variaveis(self, df):
        selected_columns = [col for col, var in self.check_vars.items() if not var.get()]
        df_sel = df.drop(selected_columns, axis=1)

        return df_sel

    def apply_kmeans(self):
        if not self.file_path:
            messagebox.showerror("Erro", "Nenhum arquivo selecionado")
            return

        try:
            n_clusters = int(self.entry_clusters.get())
            if n_clusters <= 0:
                messagebox.showerror("Erro", "O número de clusters deve ser maior que 0")
                return
        except ValueError:
            messagebox.showerror("Erro", "Por favor, insira um número válido de clusters")
            return

        try:
            depth = int(self.entry_depth.get())
            if depth < 0:
                messagebox.showerror("Erro", "A profundidade deve ser maior ou igual a 0")
                return
        except ValueError:
            messagebox.showerror("Erro", "Por favor, insira um número válido de profundidade")
            return


        try:
            df = functions.transforma_em_data_frame(self.file_path, depth)
            df_sel = self.seleciona_variaveis(df)
            df_scaled = functions.colocar_na_escala(df_sel)

            df_agrupamento = functions.agrupamento(df_scaled, n_clusters)
            df_agrupamento = df_agrupamento.reset_index()

            if self.var_save.get():
                save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
                messagebox.showinfo("Salvo", f"Imagem salva em {save_path}")
            else:
                save_path = None
            functions.scientific_map(df_agrupamento, [-41, -35, -22, -14], self.entry_title.get(), save_path)

        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro: {e}")



if __name__ == "__main__":
    root = tk.Tk()
    app = KMeansApp(root)
    root.mainloop()
